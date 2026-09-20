#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# This deliberately small source census recognizes the Rust style used by the
# two retained baseline trainers. It is evidence about these files, not a Rust
# parser or a general duplication detector.
perl - "$root/src/examples/mlp/src/baseline.rs" "$root/src/examples/cnn/src/train.rs" <<'PERL'
use strict;
use warnings;

sub functions {
    my ($path) = @_;
    open my $handle, '<', $path or die "cannot read $path: $!\n";
    local $/;
    my $source = <$handle>;
    my %result;

    while ($source =~ /^( {0,4})fn (\w+)\b/mg) {
        my ($indent, $name) = ($1, $2);
        my $start = $-[0];
        my $scope = 'host';
        if (length $indent) {
            my $prefix = substr($source, 0, $start);
            while ($prefix =~ /^(?:impl|mod) (\w+)\s*\{/mg) {
                $scope = $1;
            }
        }

        my $open = index($source, '(', $+[0]);
        die "missing parameter list for $name in $path\n" if $open < 0;
        my $depth = 1;
        my $cursor = $open + 1;
        while ($depth) {
            die "unterminated parameter list for $name in $path\n" if $cursor >= length $source;
            my $char = substr($source, $cursor++, 1);
            $depth++ if $char eq '(';
            $depth-- if $char eq ')';
        }
        my $body = index($source, '{', $cursor);
        die "missing body for $name in $path\n" if $body < 0;
        $depth = 1;
        $cursor = $body + 1;
        while ($depth) {
            die "unterminated body for $name in $path\n" if $cursor >= length $source;
            my $char = substr($source, $cursor++, 1);
            $depth++ if $char eq '{';
            $depth-- if $char eq '}';
        }

        my $text = substr($source, $start, $cursor - $start);
        my $normalized = $text;
        $normalized =~ s/\s+//g;
        my $lines = grep { /\S/ } split /\n/, $text;
        my $line = 1 + (substr($source, 0, $start) =~ tr/\n//);
        my $key = "$scope\::$name";
        die "ambiguous function key $key in $path\n" if exists $result{$key};
        $result{$key} = { normalized => $normalized, lines => $lines, start => $line };
    }
    return \%result;
}

my ($left_path, $right_path) = @ARGV;
my $left = functions($left_path);
my $right = functions($right_path);
my @same = grep {
    exists $right->{$_} && $left->{$_}{normalized} eq $right->{$_}{normalized}
} sort keys %$left;

print "| Identical function | MLP line | CNN line | Nonblank lines |\n";
print "|---|---:|---:|---:|\n";
my $total = 0;
for my $name (@same) {
    print "| `$name` | $left->{$name}{start} | $right->{$name}{start} | $left->{$name}{lines} |\n";
    $total += $left->{$name}{lines};
}
print "\n", scalar(@same), " identical definitions; $total nonblank lines per trainer.\n";
PERL
