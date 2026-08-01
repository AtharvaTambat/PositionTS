"""Train the PaiNN interatomic potential used by the NEB drivers, on the QM9x database."""
import itertools
import os
from argparse import ArgumentParser
import sys
sys.path.append("../neuralneb")

import torch

import ase.db
from neuralneb import utils
from neuralneb.painn import PaiNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LossFn(torch.nn.Module):
    def __init__(self, energy_key, forces_key, rho):
        super().__init__()
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.rho = rho

    def forward(self, batch, result):
        diff_energy = batch[self.energy_key] - result[self.energy_key]
        err_sq_energy = torch.sum(diff_energy ** 2)

        diff_forces = batch[self.forces_key] - result[self.forces_key]
        err_sq_forces = torch.sum(diff_forces ** 2) / 3

        err_sq = self.rho * err_sq_energy + (1 - self.rho) * err_sq_forces
        return err_sq


def main(args):  # pylint: disable=redefined-outer-name
    painn = PaiNN(
        num_interactions=3,
        hidden_state_size=256,
        cutoff=5,
    )
    painn.to(DEVICE)

    loss_fn = LossFn(
        energy_key="energy",
        forces_key="forces",
        rho=0.5,
    )

    optimizer = torch.optim.Adam(painn.parameters(), amsgrad=True, weight_decay=0.01)
    assert os.path.exists(
        args.db
    ), 'Dataset doesn\'t exist. If you are using default settings run "python scripts/download_qm9x.db"'

    dataset = utils.get_dataset(args.db)
    # Split dataset (90% train, 10% valid)
    n_total = len(dataset)
    n_valid = int(0.1 * n_total)
    n_train = n_total - n_valid

    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_valid],
        generator=torch.Generator().manual_seed(42)
    )
    MAX_EPOCHS = 2000
    pin_memory = DEVICE == 'cuda'
    collate_atoms = utils.CollateAtoms(pin_memory=pin_memory)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=128, collate_fn=collate_atoms
    # )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_atoms
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=64, shuffle=False, collate_fn=collate_atoms
    )
    best_val_loss = float("inf")

    print("Dataset loaded", flush=True)
    print(DEVICE, flush=True)
    print(len(train_dataset), flush=True)

    step = 0
    epoch = 0
    # for epoch in itertools.count(epoch):
    for epoch in range(epoch, MAX_EPOCHS):
        print(f'epoch {epoch} started')
        # for batch in dataloader:
        for batch in train_loader:
            batch = utils.batch_to_device(batch, DEVICE)
            step += 1
            optimizer.zero_grad()
            result = painn(batch)
            loss = loss_fn(
                batch=batch,
                result=result,
            )
            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch}, step: {step}, loss: {loss.item():.2f}', flush=True)

            # if step >= 1000000:
            #     break

        # Validation
        painn.eval()
        val_loss_total = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in valid_loader:
                batch = utils.batch_to_device(batch, DEVICE)
                result = painn(batch)
                loss = loss_fn(batch=batch, result=result)
                val_loss_total += loss.item()
                val_count += 1

        val_loss = val_loss_total / max(val_count, 1)
        print(f"Validation Loss: {val_loss:.4f}", flush=True)

        painn.train()

        # torch.save(painn, "data/painn_qm9x.sd")
        # Save last checkpoint
        torch.save(painn.state_dict(), "data/painn_qm9x_last_model.pt")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(painn.state_dict(), "data/painn_qm9x_best_model.pt")
            print("Saved new best model", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("db", nargs="?", default="data/qm9x.db")
    args = parser.parse_args()
    main(args)
