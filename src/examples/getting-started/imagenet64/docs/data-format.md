# Data boundary

The preparer consumes the NPZ representation produced for the downsampled
ImageNet study, rather than inventing another download source. It accepts
`data.npy` and `labels.npy` at the archive root or under an archive directory. The
pixel matrix must be C-order uint8 with shape `[samples, 12_288]`; channel
planes are ordered red, green, blue. Labels may be little-endian unsigned
8/16-bit or signed 32/64-bit integers and must lie in the original one-based
range `1..=1000`.

The prepared file begins with a fixed 36-byte header:

```text
"AXISIM64" | version:u32 | samples:u64 | width:u32 | height:u32 |
channels:u32 | classes:u32
```

Each record is `label:u16` followed by 12,288 pixel bytes. Integers are little
endian. Exact file-length validation makes truncated files fail at open time.
The fixed-record format remains private to this example because it exists for
large, seekable byte storage. ImageNet64 does share named vision axes and
byte-to-tensor packing with Fashion-MNIST and CIFAR-100 through the internal
`axis-vision-data` crate. This keeps storage policy local while giving the
common tensor boundary one owner.

The raw-byte ImageNet64 benchmark under the Rasat research tree has the same
image geometry but deliberately discarded labels for density modeling. It can
verify pixel decoding, but it cannot establish classifier correctness or
accuracy. A labeled source is required for this example.
