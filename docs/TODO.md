# TODO

## Hybrid SCA-NCA System (NEW)
- [x] Create hybrid module combining SCA skeleton with NCA seeding
- [x] Implement multi-seed tensor creation from skeleton positions
- [x] Support different seed extraction modes (tips, junctions, uniform)
- [ ] Train NCA model specifically for multi-seed growth
- [ ] Add skeleton-guided loss function (encourage growth along skeleton)
- [ ] Experiment with different skeleton densities and seed counts

## NCA
- [ ] Create monitor function for loss visualization
- [ ] Save intermediate results for animations
- [ ] Test different parameters (epochs, grid size, images)
- [ ] Save on interrupt (KeyboardInterrupt)

## SCA
- [ ] Optimize spatial indexing for larger masks
- [ ] Add more branching patterns