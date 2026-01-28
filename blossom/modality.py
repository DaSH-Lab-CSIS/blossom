# Standard library imports
import logging
from typing import Dict, List, Tuple, Set


class ModalityManager:
    """Manages modality assignments for clients across federated learning rounds."""

    def __init__(self, modality_dict: Dict[Tuple[str, ...], int]) -> None:
        """Initialize the modality manager."""
        self.modality_dict = modality_dict

        self.all_modalities: Set[str] = set()
        self.modality_id_list: List[int] = []
        self.client_to_modality_map: Dict[str, int] = {}

        self._tuple_to_id_map = {key: "+".join(key) for key in self.modality_dict.keys()}
        self._id_to_tuple_map = {"+".join(key): key for key in self.modality_dict.keys()}
        self._initialize_modality_id_list()

    def _initialize_modality_id_list(self) -> None:
        """Initialize the modality list based on modality_dict."""
        self.modality_id_list = []
        for key in self.modality_dict.keys():
            for modality in key:
                self.all_modalities.add(modality)
            modality_id = self.tuple_to_id(key)
            count = self.modality_dict[key]
            self.modality_id_list.extend([modality_id] * count)

    def tuple_to_id(self, modality: Tuple[str, ...]) -> int:
        """Convert modality tuple to unique ID."""
        if modality not in self._tuple_to_id_map:
            raise ValueError(f"Unknown modality tuple: {modality}")
        return self._tuple_to_id_map[modality]

    def id_to_tuple(self, modality_id: int) -> Tuple[str, ...]:
        """Convert unique ID back to modality tuple."""
        if modality_id not in self._id_to_tuple_map:
            raise ValueError(f"Unknown modality ID: {modality_id}")
        return self._id_to_tuple_map[modality_id]

    def all_assigned(self) -> bool:
        """Return whether all modalities have been assigned to clients."""
        return len(self.client_to_modality_map) >= len(self.modality_id_list)

    def get_modality_dict(self) -> Dict[Tuple[str, ...], int]:
        """Get the original modality dictionary."""
        return self.modality_dict

    def get_all_modalities(self) -> List[str]:
        """Get the list of all modality names."""
        return list(self.all_modalities)

    def get_modality(self, client_id: str) -> int:
        """Retrieve the modality configuration for a client."""
        if client_id not in self.client_to_modality_map:
            raise ValueError(f"Modality for client {client_id} not set.")
        return self.client_to_modality_map[client_id]

    def set_modality(self, client_id: str) -> None:
        """Assign a modality configuration to a client in a round-robin fashion."""
        if client_id not in self.client_to_modality_map:
            if len(self.client_to_modality_map) >= len(self.modality_id_list):
                raise ValueError("Number of clients exceeds number of available modalities.")
            
            idx = len(self.client_to_modality_map) % len(self.modality_id_list)
            self.client_to_modality_map[client_id] = self.modality_id_list[idx]

    def print_client_modality_mapping(self) -> None:
        """Print the mapping of clients to their assigned modality IDs."""
        logging.info("Client IDs (and their modalities):")
        for client_id, modality_id in self.client_to_modality_map.items():
            logging.info(f"    Client {client_id} ({modality_id})")