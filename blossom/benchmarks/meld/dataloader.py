# Standard library imports
from typing import Dict, Type, Union, List, Any, Callable

# Torch imports
import torch
import torchaudio
import numpy as np
from datasets import Dataset as HFDataset
from transformers import MobileBertModel, MobileBertTokenizer

# Local imports
from blossom.dataloader import MultimodalDataset


# ============================================================================
# Dataset Class
# ============================================================================

class MELDDataset(MultimodalDataset):
    """MELD emotion recognition dataset with audio and text modalities.
    
    Uses 4 emotion classes: anger, joy, neutral, sadness
    (as in the fed-multimodal paper)
    
    Audio features: 80-dim fbank features (matching fed-multimodal extract_audio_feature.py)
    Text features: 512-dim MobileBERT embeddings (matching fed-multimodal extract_text_feature.py)
    """

    # MELD 4 emotion classes mapping
    EMOTION_TO_INT = {
        'anger': 0,
        'joy': 1,
        'neutral': 2,
        'sadness': 3,
    }
    VALID_LABELS = {0, 1, 2, 3}
    
    # Speakers to exclude (generic/unnamed speakers)
    FILTERED_SPEAKERS = {"All", "Man", "Policeman", "Tag", "Woman"}
    
    # Audio processing parameters 
    TARGET_SR = 16000
    FRAME_LENGTH = 25  
    FRAME_SHIFT = 10   
    N_MELS = 80
    MAX_AUDIO_LEN = 1000
    
    # Text processing parameters
    MAX_TEXT_LEN = 32

    def __init__(
        self,
        dataset: HFDataset,
        train_split: bool,
    ) -> None:
        super().__init__(dataset, train_split)
        
        # Filter to only keep samples with 4 target emotions and valid speakers
        self.samples = []
        for sample in dataset:
            speaker = sample.get("speaker", "")
            if speaker in self.FILTERED_SPEAKERS:
                continue
            
            emotion = sample.get("emotion")
            if isinstance(emotion, str):
                if emotion.lower() in self.EMOTION_TO_INT:
                    self.samples.append(sample)
            elif isinstance(emotion, int):
                if emotion in self.VALID_LABELS:
                    self.samples.append(sample)
        
        # Fbank transform 
        self.fbank_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.TARGET_SR,
            n_fft=int(self.TARGET_SR * self.FRAME_LENGTH / 1000),
            hop_length=int(self.TARGET_SR * self.FRAME_SHIFT / 1000),
            n_mels=self.N_MELS,
        )
        self.resamplers = {}
        
        # Initialize MobileBERT for text feature extraction 
        self.tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
        self.text_model = MobileBertModel.from_pretrained("google/mobilebert-uncased")
        self.text_model.eval()

    def _extract_fbank_features(self, audio_data: Dict) -> torch.Tensor:
        """
        Extract fbank features from audio (matching fed-multimodal).
        Returns: (T, 80) tensor
        """
        audio_array = np.array(audio_data["array"], dtype=np.float32)
        sample_rate = audio_data["sampling_rate"]
        
        waveform = torch.from_numpy(audio_array).float()
        
        # Ensure mono
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Resample to 16kHz if needed
        if sample_rate != self.TARGET_SR:
            if sample_rate not in self.resamplers:
                self.resamplers[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate, self.TARGET_SR
                )
            waveform = self.resamplers[sample_rate](waveform)
        
        # Extract fbank features
        fbank = self.fbank_transform(waveform)  
        fbank = fbank.squeeze(0).transpose(0, 1)  
        
        # Truncate to max length 
        if fbank.shape[0] > self.MAX_AUDIO_LEN:
            fbank = fbank[:self.MAX_AUDIO_LEN]
        
        return fbank

    def _extract_text_features(self, text: str) -> torch.Tensor:
        """
        Extract MobileBERT features from text (matching fed-multimodal).
        Returns: (seq_len, 512) tensor
        """
        text_encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.MAX_TEXT_LEN,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            text_outputs = self.text_model(
                input_ids=text_encoding['input_ids'],
                attention_mask=text_encoding['attention_mask']
            )
            text_features = text_outputs.last_hidden_state.squeeze(0)
        
        return text_features

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        item = self.samples[idx]
        
        # Process audio - extract fbank features (T, 80)
        audio_data = item["audio"]
        audio = self._extract_fbank_features(audio_data)
        
        # Process text - extract MobileBERT features (seq_len, 512)
        text = item.get("text", "")
        text_features = self._extract_text_features(text)
        
        # Get emotion label
        emotion_str = item.get("emotion")
        if isinstance(emotion_str, str):
            emotion = self.EMOTION_TO_INT.get(emotion_str.lower(), 2)
        else:
            emotion = emotion_str if emotion_str in self.VALID_LABELS else 2
        
        return {
            "audio": audio,           # (T, 80) fbank features
            "text": text_features,    # (seq_len, 512) MobileBERT features
            "emotion": torch.tensor(emotion, dtype=torch.long),
        }
    
    def __len__(self):
        return len(self.samples)


# ============================================================================
# Collate Function
# ============================================================================

def meld_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for MELD batches.
    Handles variable-length audio sequences by padding.
    Returns:
        - audio: (B, max_T, 80) tensor (padded)
        - text: (B, seq_len, 512) tensor
        - emotion: (B,) tensor
    """
    # Pad audio features to max length in batch
    audio_features = [b["audio"] for b in batch]  # List of (T_i, 80) tensors
    max_audio_len = max(a.shape[0] for a in audio_features)
    
    # Pad each audio tensor to max_audio_len
    padded_audios = []
    for audio in audio_features:
        T, n_mels = audio.shape
        if T < max_audio_len:
            padding = torch.zeros(max_audio_len - T, n_mels)
            audio = torch.cat([audio, padding], dim=0)
        padded_audios.append(audio)
    
    audios = torch.stack(padded_audios)  # (B, max_T, 80)
    
    # Text: stack features (already fixed length from MobileBERT)
    text = torch.stack([item["text"] for item in batch])  # (B, seq_len, 512)
    
    # Stack emotion labels
    emotions = torch.stack([b["emotion"] for b in batch])  # (B,)
    
    return {
        "audio": audios,        # (B, max_T, 80)
        "text": text,          # (B, seq_len, 512)
        "emotion": emotions,        # (B,)
    }


# ============================================================================
# Factory Function
# ============================================================================

def get_dataset_class() -> Type:
    """Return the MELDDataset class."""
    return MELDDataset

def get_collate_fn() -> Callable:
    """Return the MELD collate function."""
    return meld_collate_fn