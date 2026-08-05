"""
Evolutionary swarm -- stage 3 of the pipeline. It replaced the particle
advection pass, which has been deleted: the swarm runs inside the timeline
instead of as a separate clip stitched on at the end.

Design and the reasoning behind every choice: docs/Evolutionary_Swarm.md.
Nothing in this package ever reads a colour from the target image; the target
enters only through `fields.compute_error`, which the agents never see.

Three layers, and it is worth knowing which one you are in:

- the **engine** -- `agents`, `selection`, `fields`, `simulation`: the design
  document, executed;
- the **judge** -- `metrics` (the population), `quality` (the picture) and
  `objective` (what "good" means, as arithmetic). Nothing here is visible to
  the engine;
- the **operator** -- `lab` compares configurations a person thought of,
  `tuning` searches for ones nobody did, `experiment`/`report`/`animate` turn
  either into artefacts. See docs/Swarm_Tuning.md.
"""
