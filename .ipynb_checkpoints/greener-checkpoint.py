







# Uses 140 bins evenly distributed near each residue
n_bins_pot = 140 

# bin centers: with this we get the centers of the bins depending on the distances

def get_bin_centres(min_dist, max_dist):
    gap_dist = (max_dist - min_dist) / n_bins_pot
    bcs_pot = [min_dist + i * gap_dist + 0.5 * gap_dist for i in range(n_bins_pot)]
    return bcs_pot[1:-1]

interactions = []
dist_bin_centres = []


# Generate distance interaction list

for i, aa_1 in enumerate(aas):
    for ai, atom_1 in enumerate(atoms):
        for atom_2 in atoms[(ai + 1):]:
            interactions.append(f"{aa_1}_{atom_1}_{aa_1}_{atom_2}_same")
            dist_bin_centres.append(get_bin_centres(0.7, 5.6))
    for aa_2 in aas[i:]:
        for ai, atom_1 in enumerate(atoms):
            atom_iter = atoms if aa_1 != aa_2 else atoms[ai:]
            for atom_2 in atom_iter:
                interactions.append(f"{aa_1}_{atom_1}_{aa_2}_{atom_2}_other")
                dist_bin_centres.append(get_bin_centres(1.0, 15.0))
    for aa_2 in aas:
        for atom_1 in atoms:
            for atom_2 in atoms:
                for ar in range(1, n_adjacent + 1):
                    interactions.append(f"{aa_1}_{atom_1}_{aa_2}_{atom_2}_adj{ar}")
                    dist_bin_centres.append(get_bin_centres(0.7, 14.7))
interactions.append("self_placeholder") # This gets zeroed out during the simulation
dist_bin_centres.append([0.0] * n_bins_force)

# We are gonna do the same for the angles and dihedrals

from math import pi
gap_ang = (pi - pi / 3) / n_bins_pot
angle_bin_centres = [pi / 3 + i * gap_ang + 0.5 * gap_ang for i in range(n_bins_pot)][1:-1]

gap_dih = (2 * pi) / n_bins_pot
# Two invisible bins on the end imitate periodicity
dih_bin_centres = [-pi + i * gap_dih - 0.5 * gap_dih for i in range(n_bins_pot + 2)][1:-1]


#################### TRAIN MODEL ######################################################### 

def train(model_filepath, device="cpu", verbosity=0):
    max_n_steps = 2_000
    learning_rate = 1e-4
    n_accumulate = 100
    
    # Start the simulator with zeros. Let's see what is each value:
        # n_bins_pot = 140 bins evenly distributed near each residue
        # interactions = 
    simulator = Simulator(
        torch.zeros(len(interactions), n_bins_pot, device=device),
        torch.zeros(len(angles), n_aas, n_bins_pot, device=device),
        torch.zeros(len(dihedrals), n_aas * len(ss_types), n_bins_pot + 2, device=device)
    )

    train_set = ProteinDataset(train_proteins, train_val_dir, device=device)
    val_set   = ProteinDataset(val_proteins  , train_val_dir, device=device)

    optimizer = torch.optim.Adam(simulator.parameters(), lr=learning_rate)

    report("Starting training", 0, verbosity)
    for ei in count(start=0, step=1):
        # After 37 epochs reset the optimiser with a lower learning rate
        if ei == 37:
            optimizer = torch.optim.Adam(simulator.parameters(), lr=learning_rate / 2)

        train_rmsds, val_rmsds = [], []
        n_steps = min(250 * ((ei // 5) + 1), max_n_steps) # Scale up n_steps over epochs
        train_inds = list(range(len(train_set)))
        val_inds   = list(range(len(val_set)))
        shuffle(train_inds)
        shuffle(val_inds)
        simulator.train()
        optimizer.zero_grad()
        for i, ni in enumerate(train_inds):
            native_coords, inters_flat, inters_ang, inters_dih, masses, seq = train_set[ni]
            coords = simulator(native_coords.unsqueeze(0), inters_flat.unsqueeze(0),
                                inters_ang.unsqueeze(0), inters_dih.unsqueeze(0), masses.unsqueeze(0),
                                seq, native_coords.unsqueeze(0), n_steps, verbosity=verbosity)
            loss, passed = rmsd(coords[0], native_coords)
            train_rmsds.append(loss.item())
            if passed:
                loss_log = torch.log(1.0 + loss)
                loss_log.backward()
            report("  Training   {:4} / {:4} - RMSD {:6.2f} over {:4} steps and {:3} residues".format(
                    i + 1, len(train_set), loss.item(), n_steps, len(seq)), 1, verbosity)
            if (i + 1) % n_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
        simulator.eval()
        with torch.no_grad():
            for i, ni in enumerate(val_inds):
                native_coords, inters_flat, inters_ang, inters_dih, masses, seq = val_set[ni]
                coords = simulator(native_coords.unsqueeze(0), inters_flat.unsqueeze(0),
                                    inters_ang.unsqueeze(0), inters_dih.unsqueeze(0), masses.unsqueeze(0),
                                    seq, native_coords.unsqueeze(0), n_steps, verbosity=verbosity)
                loss, passed = rmsd(coords[0], native_coords)
                val_rmsds.append(loss.item())
                report("  Validation {:4} / {:4} - RMSD {:6.2f} over {:4} steps and {:3} residues".format(
                        i + 1, len(val_set), loss.item(), n_steps, len(seq)), 1, verbosity)
        torch.save({"distances": simulator.ff_distances.data,
                    "angles"   : simulator.ff_angles.data,
                    "dihedrals": simulator.ff_dihedrals.data,
                    "optimizer": optimizer.state_dict()},
                    model_filepath)
        report("Epoch {:4} - med train/val RMSD {:6.3f} / {:6.3f} over {:4} steps".format(
                ei + 1, np.median(train_rmsds), np.median(val_rmsds), n_steps), 0, verbosity)