"""
Evolutionary swarm -- stage 3 of the pipeline, replaces `particles/`.

Design and the reasoning behind every choice: docs/Evolutionary_Swarm.md.
Nothing in this package ever reads a colour from the target image; the target
enters only through `fields.compute_error`, which the agents never see.
"""
