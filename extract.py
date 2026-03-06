import argparse
from pathlib import Path
import enlighten
import numpy as np
import torch
import torchaudio
import transformers
import torch.nn.functional as F

from wavlm.WavLM import WavLM, WavLMConfig

class Model():
    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.model = self.load_model()
        self.checkpoint_path = None

    def load_model(self):
        if self.model_name == 'wavlm_large':  
            checkpoint = torch.load('checkpoints/WavLM-Large.pt', map_location=self.device)
            config = WavLMConfig(checkpoint['cfg'])
            model = WavLM(config)
            model.load_state_dict(checkpoint['model'])
            model.to(self.device)
            model.eval()
            return model

        elif self.model_name == 'hubert_soft':
            model = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
            model.to(self.device)
            model.eval()
            return model

        elif self.model_name in ["hubert_large", "mhubert", "chinese_hubert_large"]:
            if self.model_name == "hubert_large":
                self.checkpoint_path = "facebook/hubert-large-ls960-ft"
            elif self.model_name == "mhubert":
                self.checkpoint_path = "utter-project/mHuBERT-147"
            elif self.model_name == "chinese_hubert_large":
                self.checkpoint_path = "TencentGameMate/chinese-hubert-large"
            model = transformers.AutoModel.from_pretrained(self.checkpoint_path)
            model.to(self.device)
            model.eval()
            return model

        else:
            raise ValueError(f"Model {self.model_name} not supported.")

    def extract_features(self, waveform, layers, sample_rate, layer_norm=True):

        if not isinstance(layers, list):
            layers = [layers]

        if self.checkpoint_path is not None:
            feature_extractor = transformers.AutoProcessor.from_pretrained(
                self.checkpoint_path
            ).to(self.device)
            inputs = feature_extractor(
                waveform.squeeze().cpu().numpy(), 
                sampling_rate=sample_rate, 
                return_tensors="pt", 
                padding=True
            ).input_values.to(self.device)
            features = self.extract_transformer_features(inputs, layers)
            return features

        waveform = self.preprocess_waveform(waveform, sample_rate, layer_norm=layer_norm)
        
        if self.model_name == 'wavlm_large':
            features = self.extract_wavlm_features(waveform, layers)
        elif self.model_name == 'hubert_soft':
            features = self.extract_hubert_soft_features(waveform, layers)
        else:
            raise ValueError(f"Model {self.model_name} not supported.")

        return features
    
    @torch.inference_mode()
    def extract_transformer_features(self, inputs, layers):

        all_features = self.model(inputs).hidden_states
        features = {
            layer: all_features[layer].squeeze().cpu().numpy() 
            for layer in layers
        }

        return features
        
    @torch.inference_mode()
    def extract_wavlm_features(self, waveform, layers): 

        _, results = self.model.extract_features(
            waveform, 
            output_layer=self.model.cfg.encoder_layers, 
            ret_layer_results=True
        )[0]
    
        all_features = [x.transpose(0, 1) for x, _ in results]
        features = {
            layer: all_features[layer].squeeze().cpu().numpy() 
            for layer in layers
        }

        return features

    @torch.inference_mode()
    def extract_hubert_soft_features(self, waveform, layers, non_units=False):

        waveform = waveform.unsqueeze(0)  

        if non_units:
            features = {
                layer: self.model.encode(waveform, layer=layer).squeeze().cpu().numpy() 
                for layer in layers
            }
            return features

        features = {}
        for layer in layers:
            feat, _ = self.model.encode(waveform, layer=layer)
            feat = self.model.proj(feat)
            features[layer] = feat.squeeze().cpu().numpy()

        return features

    def preprocess_waveform(self, waveform, sample_rate, target_sample_rate=16000, layer_norm=True):
        
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=target_sample_rate
            )
            waveform = resampler(waveform)
        
        assert waveform.ndim == 2, "Expected waveform to be a 2D tensor (channels, samples)"
        assert waveform.size(0) == 1, "Expected mono audio (1 channel)"

        if layer_norm:
            waveform = F.layer_norm(waveform, waveform.shape)
        
        if waveform.shape[-1] < 400:
            padding = (400 - waveform.shape[-1]) // 2
            waveform = F.pad(waveform, (padding, padding))

        waveform = waveform.to(self.device)

        return waveform
    
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(args.model_name, device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) / args.model_name 
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = enlighten.get_manager()
    audio_files = list(input_dir.glob(f'*{args.audio_extension}'))

    pbar = manager.counter(
        total=len(audio_files), 
        desc="Extracting", 
        unit="files", 
        color="cyan"
    )

    for audio_file in audio_files:

        expected_outputs = []
        for layer in args.layers:
            suffix = f'layer_{layer}' if args.not_layer_norm else f'layer_{layer}_layernorm'
            out_path = output_dir / suffix / audio_file.relative_to(input_dir).with_suffix('.npy')
            expected_outputs.append(out_path)

        if all(path.exists() for path in expected_outputs):
            pbar.update() 
            continue

        waveform, sample_rate = torchaudio.load(str(audio_file))
        features = model.extract_features(
            waveform, 
            args.layers, 
            sample_rate, 
            layer_norm=not args.not_layer_norm
        )

        for layer, layer_features in features.items():
            suffix = f'layer_{layer}' if args.not_layer_norm else f'layer_{layer}_layernorm'
            layer_output_dir = output_dir / suffix
            layer_output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = layer_output_dir / audio_file.relative_to(input_dir).with_suffix('.npy')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_file, layer_features)
        
        pbar.update()

    pbar.close()
    manager.stop()
    print(f"\nDone! Features saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio files using WavLM or HuBERT Soft models.")
    parser.add_argument('model_name', type=str, choices=['wavlm_large', 'hubert_soft', 'hubert_large', 'mhubert', 'chinese_hubert_large'], help="Model to use for feature extraction.")
    parser.add_argument('input_dir', type=str, help="Directory containing input .wav files.")
    parser.add_argument('output_dir', type=str, help="Directory to save extracted features.")
    parser.add_argument('layers', type=int, nargs='+', help="Layers to extract features from.")
    parser.add_argument('--audio_extension', type=str, default='.wav', help="Audio file extension to process (default: .wav).")
    parser.add_argument('--not_layer_norm', action='store_true', help="Apply layer normalization to the input waveform.")

    args = parser.parse_args()
    main(args)