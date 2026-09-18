# Contributing

Keep model-specific conditioning explicit. A change to the 5B first-latent rule
must not silently affect the 1.3B CLIP/y contract. Preserve 21-frame action
boundaries and invert labels when reversing videos.

Run the checks listed in README before submitting a change. Describe whether a
result was a CPU contract test, an inference evaluation, or a full training run.
Use relative paths in configs and examples. Do not commit weights, captured data,
local credentials, machine-specific paths or generated experiment logs.
