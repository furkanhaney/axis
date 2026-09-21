use axis::Result;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
use zip::ZipArchive;

pub const SIDE: usize = 64;
pub const CHANNELS: usize = 3;
pub const CLASSES: usize = 1_000;
pub const PIXELS: usize = SIDE * SIDE * CHANNELS;
const MAGIC: &[u8; 8] = b"AXISIM64";
const VERSION: u32 = 1;
const HEADER_BYTES: u64 = 36;
const RECORD_BYTES: u64 = 2 + PIXELS as u64;

#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub pixels: Vec<u8>,
    pub label: u16,
}

pub struct ImageNet64 {
    file: File,
    len: usize,
}

impl ImageNet64 {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let mut header = [0_u8; HEADER_BYTES as usize];
        file.read_exact(&mut header)?;
        if &header[..8] != MAGIC {
            return Err(format!("{} is not an Axis ImageNet64 file", path.display()).into());
        }
        let version = u32::from_le_bytes(header[8..12].try_into()?);
        if version != VERSION {
            return Err(format!("unsupported Axis ImageNet64 version {version}").into());
        }
        let count = u64::from_le_bytes(header[12..20].try_into()?);
        if count == 0 {
            return Err("an Axis ImageNet64 file must contain at least one sample".into());
        }
        let geometry = [
            u32::from_le_bytes(header[20..24].try_into()?),
            u32::from_le_bytes(header[24..28].try_into()?),
            u32::from_le_bytes(header[28..32].try_into()?),
            u32::from_le_bytes(header[32..36].try_into()?),
        ];
        if geometry != [SIDE as u32, SIDE as u32, CHANNELS as u32, CLASSES as u32] {
            return Err(format!("unexpected ImageNet64 geometry {geometry:?}").into());
        }
        let expected = HEADER_BYTES
            .checked_add(
                count
                    .checked_mul(RECORD_BYTES)
                    .ok_or("dataset size overflow")?,
            )
            .ok_or("dataset size overflow")?;
        let actual = file.metadata()?.len();
        if actual != expected {
            return Err(format!(
                "ImageNet64 file length mismatch: header requires {expected} bytes, found {actual}"
            )
            .into());
        }
        Ok(Self {
            file,
            len: usize::try_from(count)?,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn read(&mut self, index: usize) -> Result<Sample> {
        if index >= self.len {
            return Err(format!("sample index {index} is outside 0..{}", self.len).into());
        }
        let offset = HEADER_BYTES
            .checked_add(
                u64::try_from(index)?
                    .checked_mul(RECORD_BYTES)
                    .ok_or("sample offset overflow")?,
            )
            .ok_or("sample offset overflow")?;
        self.file.seek(SeekFrom::Start(offset))?;
        let mut label = [0_u8; 2];
        self.file.read_exact(&mut label)?;
        let label = u16::from_le_bytes(label);
        if usize::from(label) >= CLASSES {
            return Err(format!("stored class {label} is outside 0..{CLASSES}").into());
        }
        let mut pixels = vec![0; PIXELS];
        self.file.read_exact(&mut pixels)?;
        Ok(Sample { pixels, label })
    }
}

#[derive(Debug)]
struct NpyHeader {
    dtype: String,
    fortran: bool,
    shape: Vec<usize>,
}

fn read_npy_header(reader: &mut impl Read) -> Result<NpyHeader> {
    let mut prefix = [0_u8; 8];
    reader.read_exact(&mut prefix)?;
    if &prefix[..6] != b"\x93NUMPY" {
        return Err("NPZ member is not an NPY array".into());
    }
    let header_len = match (prefix[6], prefix[7]) {
        (1, 0) => {
            let mut bytes = [0_u8; 2];
            reader.read_exact(&mut bytes)?;
            usize::from(u16::from_le_bytes(bytes))
        }
        (2 | 3, 0) => {
            let mut bytes = [0_u8; 4];
            reader.read_exact(&mut bytes)?;
            usize::try_from(u32::from_le_bytes(bytes))?
        }
        version => return Err(format!("unsupported NPY version {version:?}").into()),
    };
    if header_len > 1 << 20 {
        return Err("NPY header exceeds 1 MiB".into());
    }
    let mut bytes = vec![0; header_len];
    reader.read_exact(&mut bytes)?;
    let text = std::str::from_utf8(&bytes)?;
    let dtype = quoted_field(text, "descr")?;
    let fortran = match scalar_field(text, "fortran_order")? {
        "False" => false,
        "True" => true,
        value => return Err(format!("invalid NPY fortran_order {value:?}").into()),
    };
    let shape_text = scalar_field(text, "shape")?;
    let shape_text = shape_text
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
        .ok_or("invalid NPY shape tuple")?;
    let shape = shape_text
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::parse)
        .collect::<std::result::Result<Vec<usize>, _>>()?;
    if shape.is_empty() {
        return Err("scalar NPY arrays are not supported".into());
    }
    Ok(NpyHeader {
        dtype,
        fortran,
        shape,
    })
}

fn field_tail<'a>(text: &'a str, key: &str) -> Result<&'a str> {
    let single = format!("'{key}'");
    let double = format!("\"{key}\"");
    let start = text
        .find(&single)
        .map(|index| index + single.len())
        .or_else(|| text.find(&double).map(|index| index + double.len()))
        .ok_or_else(|| format!("NPY header is missing {key:?}"))?;
    let tail = text[start..]
        .strip_prefix(':')
        .or_else(|| text[start..].trim_start().strip_prefix(':'))
        .ok_or_else(|| format!("NPY header field {key:?} has no colon"))?;
    Ok(tail.trim_start())
}

fn quoted_field(text: &str, key: &str) -> Result<String> {
    let tail = field_tail(text, key)?;
    let quote = tail.chars().next().ok_or("truncated NPY header")?;
    if quote != '\'' && quote != '"' {
        return Err(format!("NPY header field {key:?} is not quoted").into());
    }
    let value = &tail[quote.len_utf8()..];
    let end = value.find(quote).ok_or("unterminated NPY string")?;
    Ok(value[..end].to_owned())
}

fn scalar_field<'a>(text: &'a str, key: &str) -> Result<&'a str> {
    let tail = field_tail(text, key)?;
    let end = if tail.starts_with('(') {
        tail.find(')').map(|index| index + 1)
    } else {
        tail.find(',')
    }
    .ok_or_else(|| format!("unterminated NPY field {key:?}"))?;
    Ok(tail[..end].trim())
}

fn member_name(archive: &mut ZipArchive<File>, stem: &str) -> Result<String> {
    let suffix = format!("{stem}.npy");
    let matches: Vec<_> = archive
        .file_names()
        .filter(|name| *name == suffix || name.ends_with(&format!("/{suffix}")))
        .map(str::to_owned)
        .collect();
    match matches.as_slice() {
        [name] => Ok(name.clone()),
        [] => Err(format!("NPZ archive is missing {suffix}").into()),
        _ => Err(format!("NPZ archive contains multiple {suffix} members").into()),
    }
}

fn read_labels(path: &Path) -> Result<Vec<u16>> {
    let mut archive = ZipArchive::new(File::open(path)?)?;
    let name = member_name(&mut archive, "labels")?;
    let mut member = archive.by_name(&name)?;
    let header = read_npy_header(&mut member)?;
    if header.fortran || header.shape.len() != 1 {
        return Err("ImageNet64 labels must be a C-order rank-1 NPY array".into());
    }
    let width = match header.dtype.as_str() {
        "|u1" | "<u1" => 1,
        "<u2" => 2,
        "<i4" => 4,
        "<i8" => 8,
        dtype => return Err(format!("unsupported ImageNet64 label dtype {dtype:?}").into()),
    };
    let bytes_len = header.shape[0]
        .checked_mul(width)
        .ok_or("label array size overflow")?;
    let mut bytes = vec![0; bytes_len];
    member.read_exact(&mut bytes)?;
    let mut labels = Vec::with_capacity(header.shape[0]);
    for chunk in bytes.chunks_exact(width) {
        let one_based = match width {
            1 => u64::from(chunk[0]),
            2 => u64::from(u16::from_le_bytes(chunk.try_into()?)),
            4 => u64::try_from(i32::from_le_bytes(chunk.try_into()?))?,
            8 => u64::try_from(i64::from_le_bytes(chunk.try_into()?))?,
            _ => unreachable!(),
        };
        if !(1..=CLASSES as u64).contains(&one_based) {
            return Err(format!("ImageNet64 label {one_based} is outside 1..={CLASSES}").into());
        }
        labels.push(u16::try_from(one_based - 1)?);
    }
    let mut trailing = [0_u8; 1];
    if member.read(&mut trailing)? != 0 {
        return Err("ImageNet64 labels array has trailing data".into());
    }
    Ok(labels)
}

fn copy_images(path: &Path, labels: &[u16], output: &mut impl Write) -> Result<()> {
    let mut archive = ZipArchive::new(File::open(path)?)?;
    let name = member_name(&mut archive, "data")?;
    let mut member = archive.by_name(&name)?;
    let header = read_npy_header(&mut member)?;
    if header.fortran || header.dtype != "|u1" && header.dtype != "<u1" {
        return Err("ImageNet64 pixels must be a C-order uint8 NPY array".into());
    }
    if header.shape != [labels.len(), PIXELS] {
        return Err(format!(
            "ImageNet64 data shape {:?} does not match {} labels and {PIXELS} pixels",
            header.shape,
            labels.len()
        )
        .into());
    }
    let mut pixels = vec![0; PIXELS];
    for &label in labels {
        member.read_exact(&mut pixels)?;
        output.write_all(&label.to_le_bytes())?;
        output.write_all(&pixels)?;
    }
    let mut trailing = [0_u8; 1];
    if member.read(&mut trailing)? != 0 {
        return Err("ImageNet64 pixel array has trailing data".into());
    }
    Ok(())
}

pub fn prepare_npz_shards(paths: &[PathBuf], output: impl AsRef<Path>) -> Result<usize> {
    if paths.is_empty() {
        return Err("at least one ImageNet64 NPZ shard is required".into());
    }
    let labels = paths
        .iter()
        .map(|path| read_labels(path))
        .collect::<Result<Vec<_>>>()?;
    let count = labels.iter().try_fold(0_usize, |total, shard| {
        total
            .checked_add(shard.len())
            .ok_or("sample count overflow")
    })?;
    if count == 0 {
        return Err("ImageNet64 NPZ shards contain no samples".into());
    }
    let output = output.as_ref();
    let temporary = output.with_extension("axis-imagenet64.tmp");
    let result = (|| -> Result<()> {
        let mut writer = File::create(&temporary)?;
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&u64::try_from(count)?.to_le_bytes())?;
        for value in [SIDE as u32, SIDE as u32, CHANNELS as u32, CLASSES as u32] {
            writer.write_all(&value.to_le_bytes())?;
        }
        for (path, labels) in paths.iter().zip(&labels) {
            copy_images(path, labels, &mut writer)?;
        }
        writer.sync_all()?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    std::fs::rename(temporary, output)?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::Cursor,
        sync::atomic::{AtomicU64, Ordering},
    };
    use zip::{ZipWriter, write::SimpleFileOptions};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn npy(dtype: &str, shape: &str, body: &[u8]) -> Vec<u8> {
        let mut header =
            format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({shape}), }}");
        let prefix = 10;
        let padding = (64 - (prefix + header.len() + 1) % 64) % 64;
        header.push_str(&" ".repeat(padding));
        header.push('\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend(u16::try_from(header.len()).unwrap().to_le_bytes());
        bytes.extend(header.as_bytes());
        bytes.extend(body);
        bytes
    }

    fn fixture(data: &[u8], labels: &[i64]) -> Vec<u8> {
        let mut output = Cursor::new(Vec::new());
        {
            let mut archive = ZipWriter::new(&mut output);
            let options = SimpleFileOptions::default();
            archive.start_file("data.npy", options).unwrap();
            archive
                .write_all(&npy("|u1", &format!("{}, {PIXELS}", labels.len()), data))
                .unwrap();
            archive.start_file("labels.npy", options).unwrap();
            let raw: Vec<_> = labels
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            archive
                .write_all(&npy("<i8", &format!("{},", labels.len()), &raw))
                .unwrap();
            archive.finish().unwrap();
        }
        output.into_inner()
    }

    fn temporary(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "axis-imagenet64-{}-{}-{name}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn prepares_official_npz_layout_and_reads_random_records() -> Result<()> {
        let source = temporary("source.npz");
        let output = temporary("prepared.bin");
        let first: Vec<_> = (0..PIXELS).map(|index| index as u8).collect();
        let second: Vec<_> = (0..PIXELS).map(|index| (255 - index % 256) as u8).collect();
        let mut data = first.clone();
        data.extend(&second);
        std::fs::write(&source, fixture(&data, &[1, 1_000]))?;
        assert_eq!(
            prepare_npz_shards(std::slice::from_ref(&source), &output)?,
            2
        );
        let mut dataset = ImageNet64::open(&output)?;
        assert_eq!(dataset.len(), 2);
        assert_eq!(
            dataset.read(1)?,
            Sample {
                pixels: second,
                label: 999
            }
        );
        assert_eq!(
            dataset.read(0)?,
            Sample {
                pixels: first,
                label: 0
            }
        );
        assert!(dataset.read(2).is_err());
        std::fs::remove_file(source)?;
        std::fs::remove_file(output)?;
        Ok(())
    }

    #[test]
    fn rejects_out_of_range_labels_without_leaving_partial_output() -> Result<()> {
        let source = temporary("bad-source.npz");
        let output = temporary("bad-output.bin");
        std::fs::write(&source, fixture(&vec![0; PIXELS], &[0]))?;
        assert!(prepare_npz_shards(std::slice::from_ref(&source), &output).is_err());
        assert!(!output.exists());
        assert!(!output.with_extension("axis-imagenet64.tmp").exists());
        std::fs::remove_file(source)?;
        Ok(())
    }

    #[test]
    fn rejects_truncated_prepared_files_at_open() -> Result<()> {
        let source = temporary("truncated-source.npz");
        let output = temporary("truncated.bin");
        std::fs::write(&source, fixture(&vec![7; PIXELS], &[7]))?;
        prepare_npz_shards(std::slice::from_ref(&source), &output)?;
        let file = File::options().write(true).open(&output)?;
        file.set_len(file.metadata()?.len() - 1)?;
        assert!(ImageNet64::open(&output).is_err());
        std::fs::remove_file(source)?;
        std::fs::remove_file(output)?;
        Ok(())
    }
}
