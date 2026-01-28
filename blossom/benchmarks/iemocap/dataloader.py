from typing import *
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

class IEMOCAPDataset(MultimodalDataset):
    """IEMOCAP emotion recognition dataset with audio and text modalities."""

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

        self.int_to_label = {
            0: 'xxx', # 2507
            1: 'sad', # 1084
            2: 'sur',# 107
            3: 'dis', # 2
            4: 'fea', # 40
            5: 'fru', # 1849
            6: 'neu', # 1708
            7: 'ang', # 1103
            8: 'hap', # 595
            9: 'oth', # 3
            10: 'exc' # 1041
        }

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
        
        self.samples = []
        self.keep_labels = ["ang", "hap", "exc", "neu", "sad"]
        for sample in dataset:
            if(self.int_to_label[sample["label"]] in self.keep_labels):
                self.samples.append(sample)
    
    def _remap_labels(self, label):
        remap_dict = {
            1:0, # sad
            6:1, # neu
            7:2, # ang
            8:3, # hap
            10:3 # exc
        }
        return remap_dict[label]

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
        fbank = self.fbank_transform(waveform)  # (1, n_mels, T)
        fbank = fbank.squeeze(0).transpose(0, 1)  # (T, n_mels)
        
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

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        item = self.samples[idx]
        
        audio_data = item["audio"]
        audio_features = self._extract_fbank_features(audio_data)
        
        text = item.get("text", item.get("transcription_text", ""))
        text_features = self._extract_text_features(text)
        
        label = self._remap_labels(item['label'])
        
        return {
            "audio": audio_features,  
            "text": text_features,
            "label": torch.tensor(label, dtype=torch.long), 
        }
    
    def __len__(self):
        return len(self.samples) 


# ============================================================================
# Collate Function
# ============================================================================

def iemocap_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for IEMOCAP batches.
    Returns:
        - audio: (B, n_mfcc) tensor
        - text: list of strings
        - label: (B,) tensor
    """
    audio_features = [b["audio"] for b in batch]  
    max_audio_len = max(a.shape[0] for a in audio_features)
    
    padded_audios = []
    for audio in audio_features:
        T, n_mels = audio.shape
        if T < max_audio_len:
            padding = torch.zeros(max_audio_len - T, n_mels)
            audio = torch.cat([audio, padding], dim=0)
        padded_audios.append(audio)
    
    audios = torch.stack(padded_audios)  
    
    text = torch.stack([item["text"] for item in batch])  
    
    labels = torch.stack([b["label"] for b in batch])
    
    return {
        "audio": audios,        
        "text": text,          
        "label": labels,        
    }


# ============================================================================
# Factory Function
# ============================================================================

def get_dataset_class() -> Type:
    """Return the IEMOCAPDataset class."""
    return IEMOCAPDataset

def get_collate_fn() -> Callable:
    """Return the IEMOCAP collate function."""
    return iemocap_collate_fn