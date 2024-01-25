import numpy as np
import pandas as pd
import sys
import warnings
from itertools import permutations
from copy import deepcopy
import torch
from rdkit import Chem
from rdkit.Chem.AllChem import ReactionFromSmarts
from torch.utils.data import Dataset
from typing import Iterable, Iterator, List, Optional, Tuple

from cslvae.utils.other_utils import flatten_list, int2mix
from cslvae.utils.torch_utils import build_library_indexes, convert_1d_to_2d_indexes
from cslvae.data import TorchMol, PackedTorchMol, NUM_NODE_FEATURES, NUM_EDGE_FEATURES


FALLBACK_PRODUCT = Chem.MolFromSmiles("B1BBB1")


class CSLDataset(Dataset):
    """A dataset for the CSLVAE model.

    TODO: Complete
    """

    products_with_errors: List[int] = []

    def __init__(
        self,
        reaction_smarts_path: str,
        synthon_smiles_path: str,
        use_explicit_h: bool = False,
    ):
        # TODO: Split this function into smaller sub-functions for readability

        # If True, then add explicit hydrogens to the molecules during reactions
        self.use_explicit_h = use_explicit_h

        # Load reaction SMARTS and synthon SMILES dataframes
        reaction_df = pd.read_csv(reaction_smarts_path, sep="\s+")
        synthon_df = pd.read_csv(synthon_smiles_path, sep="\s+")

        # Map the original reaction_ids to the internal (normalized) reaction_ids
        reaction_df["orig_reaction_id"] = reaction_df[
            "reaction_id"
        ]  # Just copies the column
        reaction_df["reaction_smarts"] = reaction_df[
            "smarts"
        ]  # Just copies the column again
        orig_reaction_mapper = {
            name: i for i, name in enumerate(reaction_df["reaction_id"])
        }  # Converts names to indexes
        reaction_df["reaction_id"] = reaction_df["orig_reaction_id"].apply(
            lambda x: orig_reaction_mapper[x]
        )  # Converts to a array of integers each corresponding to a reaction index

        # Map the original synthon_ids to the internal (normalized) synthon_ids
        synthon_df["orig_synthon_id"] = synthon_df["synthon_id"]  # Again copy of column
        synthon_df["synthon_smiles"] = synthon_df["smiles"]  # Again, copy of column
        synthon_mapper = {
            smi: i
            for i, smi in enumerate(
                sorted(synthon_df["synthon_smiles"].unique())
            )  # Converts names to indexes, keeping only unique smiles
        }
        synthon_df["synthon_id"] = synthon_df["synthon_smiles"].apply(
            lambda x: synthon_mapper[x]
        )  # Now array of integers corresponding to index of unique smiles
        synthon_df["reaction_id"] = synthon_df["reaction_id"].apply(
            lambda x: orig_reaction_mapper[
                x
            ]  # Another array of integers corresponding to the reaction ID for that synthon. NOTE: What about synthons with multiple reactions?
        )

        # Form dicts for mapping internal reaction/synthon IDs to originals
        self._orig_reaction_id_lookup = {
            reaction_id: orig_reaction_id
            for reaction_id, orig_reaction_id in zip(
                reaction_df.reaction_id,
                reaction_df.orig_reaction_id,
            )
        }
        self._orig_synthon_id_lookup = {
            (reaction_id, synthon_id): orig_synthon_id
            for reaction_id, synthon_id, orig_synthon_id in zip(
                synthon_df.reaction_id,
                synthon_df.synthon_id,
                synthon_df.orig_synthon_id,
            )
        }

        # Form libtree; this is a list-of-list-of-list-of-ints. The top level corresponds to
        # reactions (i.e., len(libtree) == n_reactions), with the subsequent level corresponding
        # to the reaction R-groups, and the leaves are the normalized synthon_ids (ints) for the
        # synthons contained in the given R-group
        #
        # For example, if the zeroth reaction has two R-groups (combines two synthons),
        # then len(libtree[0]) == 2 where the first element here is the list of all
        # synthon_ids which contain the first R-group, and the second element is the
        # list of all synthon_ids which contain the second R-group.
        #
        # If a reaction has three total R-groups, then len(libtree[0]) == 3, where each
        # element in this list is the list of all synthon_ids which contain the given
        # R-group for the reaction.
        libtree = synthon_df[["reaction_id", "rgroup", "synthon_id"]]
        libtree = libtree.drop_duplicates()
        libtree = libtree.sort_values(
            by=["reaction_id", "rgroup", "synthon_id"]
        ).reset_index(drop=True)
        libtree = libtree.groupby("reaction_id")
        libtree = libtree.apply(
            lambda x: x.groupby("rgroup").apply(lambda x: x["synthon_id"].tolist())
        ).T
        libtree: List[List[List[int]]] = [
            libtree[i].tolist() for i in range(len(orig_reaction_mapper))
        ]

        assert len(set(orig_reaction_mapper.values())) == len(
            set(orig_reaction_mapper.keys())
        )
        tmp = synthon_df[["orig_synthon_id", "synthon_smiles"]].drop_duplicates()
        assert len(tmp.orig_synthon_id.unique()) == len(tmp)
        orig_synthon_mapper = {
            k: synthon_mapper[v]
            for k, v in zip(tmp.orig_synthon_id, tmp.synthon_smiles)
        }

        # Retain only specific columns
        reaction_df = reaction_df[
            ["reaction_id", "orig_reaction_id", "reaction_smarts"]
        ].sort_values(by="reaction_id")
        synthon_df = synthon_df[
            ["synthon_id", "orig_synthon_id", "synthon_smiles"]
        ].sort_values(by="synthon_id")

        # Create a bunch of attributes that will be utilized by CSLDataset's methods
        self.reaction_df: pd.DataFrame = reaction_df.drop_duplicates().reset_index(
            drop=True
        )
        self.synthon_df: pd.DataFrame = synthon_df.drop_duplicates().reset_index(
            drop=True
        )
        self.libtree: List[List[List[int]]] = libtree
        self.reaction_smarts: List[str] = (
            self.reaction_df[["reaction_id", "reaction_smarts"]]
            .drop_duplicates()["reaction_smarts"]
            .tolist()
        )
        self.synthon_smiles: List[str] = (
            self.synthon_df[["synthon_id", "synthon_smiles"]]
            .drop_duplicates()["synthon_smiles"]
            .tolist()
        )

        # The attribute _rgroup_counts is a list for every reaction where each entry for
        # a reaction counts the number of synthons for that reaction for each R-group.
        #
        # For example, if a reaction combines two synthons and there are 20 synthons
        # with the first R-group and 30 synthons with the second R-group, then the entry
        # for that reaction will be [20, 30].
        self._rgroup_counts = [
            [len(x) for x in rxn] for idx, rxn in enumerate(self.libtree)
        ]

        # The attribute _reaction_counts is a list of the number of products for each
        # reaction calculated by multiplying the total number of synthons which could
        # react for that reaction (calculated above).
        self._reaction_counts = np.array(
            [np.prod(self._rgroup_counts[k]) for k in range(self.num_reactions)]
        )

        # Cumulative sum of the number of products for each reaction (1d array with
        # shape num_reactions + 1)
        self._reaction_counts_cum = np.insert(np.cumsum(self._reaction_counts), 0, 0)

        self._orig_synthon_mapper = orig_synthon_mapper
        self._orig_reaction_mapper = orig_reaction_mapper

        self._num_products = sum(
            [np.prod([len(synthon_set) for synthon_set in rxn]) for rxn in self.libtree]
        )
        self._num_rgroups = sum([len(x) for x in self._rgroup_counts])

    def get_internal_synthon_id(self, orig_synthon_id) -> int:
        return self._orig_synthon_mapper[orig_synthon_id]

    def get_internal_reaction_id(self, orig_reaction_id) -> int:
        return self._orig_reaction_mapper[orig_reaction_id]

    @property
    def num_node_features(self) -> int:
        return NUM_NODE_FEATURES

    @property
    def num_edge_features(self) -> int:
        return NUM_EDGE_FEATURES

    @property
    def num_reactions(self) -> int:
        return len(self.reaction_smarts)

    @property
    def num_rgroups(self) -> int:
        return self._num_rgroups

    @property
    def num_synthons(self) -> int:
        return len(self.synthon_smiles)

    @property
    def num_products(self) -> int:
        return self._num_products

    def get_product_ids_by_reaction_id(self, reaction_id: int) -> Iterable[int]:
        """Returns the all the possible product IDs for a given reaction ID. Each
        product's ID is an integer in the range [0, num_products), and product IDs
        appear in the order which they are produced. For example, products from the
        first reaction will have IDs lower than products from the second reaction.

        Arguments:
            (int) reaction_id: The reaction ID for which to return the product IDs.
        """
        return range(
            self._reaction_counts_cum[reaction_id],
            self._reaction_counts_cum[reaction_id + 1],
        )

    def product2smiles(self, reaction_id: int, synthon_ids: Tuple[int, ...]) -> str:
        """Given a reaction_id and the synthon_ids to try and react, return the SMILES
        string for the product molecule.

        Arguments:
            (int) reaction_id: The reaction ID.
            (tuple) synthon_ids: A tuple of the synthon IDs to react.
        """
        return Chem.MolToSmiles(self.product2mol(reaction_id, synthon_ids))

    def _run_reaction_all_permutations(
        self,
        reaction_id: int,
        synthon_ids: Tuple[int, ...],
        reaction: Chem.rdChemReactions.ChemicalReaction,
        synthons: Tuple[Chem.rdchem.Mol, ...],
        # fallback_product: Chem.rdchem.Mol = FALLBACK_PRODUCT,
    ) -> Chem.rdchem.Mol:
        synthon_perm = list(permutations(synthons))

        # Iterate over all permutations of synthons and attempt to produce a product
        for _synthons in synthon_perm:
            product = reaction.RunReactants(_synthons)
            if len(product) > 0:
                break

        # No products present after all permutations
        if len(product) == 0:
            warnings.warn("Reaction did not produce any products.")
            self.products_with_errors.append(self.product2key(reaction_id, synthon_ids))

            return deepcopy(FALLBACK_PRODUCT)

        return product[0][0]

    def product2mol(
        self, reaction_id: int, synthon_ids: Tuple[int, ...]
    ) -> Chem.rdchem.Mol:
        """Given a reaction_id and the synthon_ids to try and react, return a RDkit Mol
        object representing the product molecule. Ordering of synthons in reaction
        arguments sometimes matters (quirk of RDkit), so all permutations are checked
        until a product is generated warning if not and returning an empty molecule.
        Additionally, the product ID is added to a list of tracked products with errors.

        Arguments:
            (int) reaction_id: The reaction ID.
            (tuple) synthon_ids: A tuple of the synthon IDs to react.
        """
        reaction = self.reaction2rxn(reaction_id)
        synthons = tuple(self.synthon2mol(i) for i in synthon_ids)
        product = reaction.RunReactants(synthons)

        # product = self._run_reaction_all_permutations(
        #     reaction_id=reaction_id,
        #     synthon_ids=synthon_ids,
        #     reaction=reaction,
        #     synthons=synthons,
        # )

        # Check if no products were produced
        if len(product) == 0:
            warnings.warn("Reaction did not produce any products.")
            self.products_with_errors.append(self.product2key(reaction_id, synthon_ids))

            return deepcopy(FALLBACK_PRODUCT)

        # Check if the product is chemically valid. Sanitization enum is 0 if
        # no errors happen during sanitization.
        if Chem.SanitizeMol(product[0][0], catchErrors=True) != 0:
            warnings.warn("Reaction produced an invalid product.")
            self.products_with_errors.append(self.product2key(reaction_id, synthon_ids))

            return deepcopy(FALLBACK_PRODUCT)

        return product[0][0]

    def product2key(
        self,
        reaction_id: int,
        synthon_ids: Tuple[int, ...],
    ) -> Tuple[str, Tuple[str, ...]]:
        orig_reaction_id = self._orig_reaction_id_lookup[reaction_id]
        orig_synthon_ids = []

        for synthon_id in synthon_ids:
            orig_synthon_id = self._orig_synthon_id_lookup[(reaction_id, synthon_id)]
            orig_synthon_ids.append(str(orig_synthon_id))

        return str(orig_reaction_id), tuple(orig_synthon_ids)

    def reaction2smarts(self, reaction_id: int) -> str:
        return self.reaction_smarts[reaction_id]

    def reaction2rxn(self, reaction_id: int) -> Chem.rdChemReactions.ChemicalReaction:
        return ReactionFromSmarts(self.reaction2smarts(reaction_id))

    def synthon2smiles(self, synthon_id: int) -> str:
        return self.synthon_smiles[synthon_id]

    def synthon2mol(self, synthon_id: int) -> Chem.rdchem.Mol:
        mol = Chem.MolFromSmiles(self.synthon2smiles(synthon_id))

        if self.use_explicit_h:
            mol = Chem.AddHs(mol)

        return mol

    def __len__(self) -> int:
        return self.num_products

    def __getitem__(self, product_id: int):
        assert 0 <= product_id < len(self)
        try:
            # Find reaction by testing which range the product_id falls into
            reaction_id = (self._reaction_counts_cum <= product_id).sum() - 1

            # How far away the desired product_id is from all products for this reaction
            product_delta = product_id - self._reaction_counts_cum[reaction_id]

            # How many synthons are in each R-group for this reaction, then find the
            # synthons which produce the desired product
            r_groups_size = self._rgroup_counts[reaction_id]
            mix = int2mix(product_delta, r_groups_size)
            synthon_ids = tuple(x[i] for i, x in zip(mix, self.libtree[reaction_id]))

            # Get reaction product and package into TorchMol
            product = self.product2smiles(reaction_id, synthon_ids)
            product = TorchMol(product)

        except Exception as e:
            print("An error occurred:", type(e), e)
            self.products_with_errors.append(self.product2key(reaction_id, synthon_ids))

            product = deepcopy(FALLBACK_PRODUCT)
            product = TorchMol(product)

        return {
            "product_id": product_id,
            "reaction_id": reaction_id,
            "synthon_ids": synthon_ids,
            "product": product,
            "synthons": [TorchMol(self.synthon2smiles(i)) for i in synthon_ids],
        }

    def create_dataloader(
        self,
        products_per_reaction: int,
        num_reactions: int = 1,
        max_iterations: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_sampler=self.create_batch_sampler(
                products_per_reaction,
                num_reactions,
                max_iterations,
            ),
            collate_fn=self.collate_fn,
        )

    def create_batch_sampler(
        self,
        products_per_reaction: int,
        num_reactions: int = 1,
        max_iterations: Optional[int] = None,
    ):
        class ReactionBatchSampler:
            def __init__(
                self_, dataset, products_per_reaction, num_reactions, max_iterations
            ):
                if num_reactions >= dataset.num_reactions:
                    raise ValueError(
                        "Cannot sample more reactions than exist in dataset."
                    )

                self_.dataset = dataset
                self_.products_per_reaction = int(products_per_reaction)
                self_.num_reactions = int(num_reactions)
                self_.max_iterations = int(max_iterations or sys.maxsize)
                self_.cum_iterations = 0

            def __iter__(self_) -> Iterator[List[int]]:
                self_.cum_iterations = 0
                while self_.cum_iterations < self_.max_iterations:
                    self_.cum_iterations += 1
                    reaction_ids = np.random.choice(
                        range(self_.dataset.num_reactions),
                        self_.num_reactions,
                        replace=False,
                    ).tolist()
                    ranges = [
                        self_.dataset.get_product_ids_by_reaction_id(i)
                        for i in reaction_ids
                    ]
                    fn = lambda rng: np.random.randint(
                        rng.start, rng.stop, (self_.products_per_reaction,)
                    )
                    indexes = flatten_list([fn(rng).tolist() for rng in ranges])
                    yield indexes

        return ReactionBatchSampler(
            self, products_per_reaction, num_reactions, max_iterations
        )

    @staticmethod
    def collate_fn(items):
        items = [item for item in items if item is not None]
        libtree = {}
        for item in items:
            (reaction_id, synthon_ids) = (item["reaction_id"], item["synthon_ids"])
            if reaction_id not in libtree:
                libtree[reaction_id] = [{s} for s in synthon_ids]
            else:
                for i in range(len(libtree[reaction_id])):
                    libtree[reaction_id][i].update({synthon_ids[i]})

        orig_reaction_ids = sorted(libtree.keys())
        libtree = [list(map(sorted, libtree[k])) for k in orig_reaction_ids]

        library_indexes = build_library_indexes(libtree)

        reaction_mapper = {j: i for i, j in enumerate(orig_reaction_ids)}
        synthon_mapper = {
            j.item(): i for i, j in enumerate(library_indexes["orig_synthon_ids"])
        }

        product2reaction: List[int] = [
            reaction_mapper[item["reaction_id"]] for item in items
        ]
        block2product: List[int] = flatten_list(
            [len(item["synthon_ids"]) * [i] for i, item in enumerate(items)]
        )
        block2rgroup: List[int] = flatten_list(
            [list(range(len(item["synthon_ids"]))) for i, item in enumerate(items)]
        )
        block2synthon: List[int] = flatten_list(
            [[synthon_mapper[i] for i in item["synthon_ids"]] for item in items]
        )

        product2reaction = convert_1d_to_2d_indexes(torch.tensor(product2reaction))
        block2product = convert_1d_to_2d_indexes(torch.tensor(block2product))
        block2reaction = convert_1d_to_2d_indexes(product2reaction[1][block2product[1]])
        block2rgroup = convert_1d_to_2d_indexes(
            torch.tensor(block2rgroup)
            + library_indexes["first_rgroup_by_reaction"][block2reaction[1]]
        )
        block2synthon = convert_1d_to_2d_indexes(torch.tensor(block2synthon))

        (idx0, idx1) = block2synthon[:, block2synthon[1].argsort()]
        idx2 = idx0[
            torch.where(torch.nn.functional.pad(idx1.diff(), (1, 0), value=1))[0]
        ]

        products = [item["product"] for item in items]
        blocks = flatten_list([item["synthons"] for item in items])
        synthons = [blocks[i.item()] for i in idx2]

        return {
            "library_indexes": library_indexes,
            "product2reaction": product2reaction,
            "block2product": block2product,
            "block2reaction": block2reaction,
            "block2rgroup": block2rgroup,
            "block2synthon": block2synthon,
            "products": PackedTorchMol(products),
            "synthons": PackedTorchMol(synthons),
        }
