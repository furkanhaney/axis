# Migration notes

## Scientific contract represented

The source study asks whether physical and technical capacity predicts real
PPP GDP across countries and time. This consumer keeps the representation used
by the current panel protocol:

- the target is `log10(real_ppp_gdp_2021_intl_usd)`;
- electricity, oil, electrical/electronic trades workers, and electronics
  papers use the source validity rules and log transforms;
- invalid or unavailable inputs are imputed with means measured on training
  rows, and four explicit missingness flags retain the missing-data pattern;
- `year - 2000` enters as a linear input;
- all nine inputs and the target are standardized using training rows only;
- the model is a one-hidden-layer ReLU MLP with a linear scalar output;
- validation targets never affect preprocessing or gradient updates.

The executable exposes two deliberately named smoke splits. `country` holds
out whole countries, so no ISO3 code occurs on both sides. `future` trains on
years through 2019 and validates on 2020--2024 using contemporaneous inputs.
The smoke caps each side after constructing the split; it never randomly mixes
country-years and then calls the result a country or time evaluation.

## What this witness does not preserve

This is not the completed panel fit. It does not reproduce the five frozen
outer country folds, the three predeclared inner splits, the width family, the
no-time and ridge comparisons, the fresh outer refits, persistence baseline,
artifact serialization, or published metrics. Its deterministic country split
is only a small mechanics fixture.

The witness uses the protocol's AdamW learning rate `0.01` and decoupled weight
decay `0.001`; parameter values and optimizer moments remain device-resident.
It preserves validation every ten steps, best-checkpoint selection, and
restoration. The bounded default runs 40 steps rather than the protocol's
400-step ceiling and 60-step patience.

Passing this program establishes that the framework can express the data path,
regression loss, split invariant, and checkpoint mechanics. It is not evidence
for GDP predictive performance and must not enter the study's result tables.
