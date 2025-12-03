from omegaconf import DictConfig
from lightning.pytorch import LightningDataModule
import torchvision.transforms as T # type: ignore
import torchaudio.transforms as T_audio # type: ignore
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import os
from torch.distributions import Beta
import pandas as pd
import numpy as np
from torchcodec.decoders import AudioDecoder
import torch.nn.functional as F

import nvtx

def load_slice_with_torchcodec(audio_path: str, loc: float, window: float):
    """
    audio_path : chemin du fichier audio
    loc        : position cible en secondes (comme dans ton code actuel)
    window     : demi-fenêtre en secondes (même sémantique que ton self.window)
    """
    # 1) Décoder tout le fichier (équivalent à torchaudio.load)
    dec = AudioDecoder(audio_path)
    samples = dec.get_all_samples()   # AudioSamples
    waveform = samples.data        # torch.Tensor [channels, time]
    sample_rate = samples.sample_rate # int

    # 2) total_frames (équivalent à torchaudio.info(...).num_frames)
    total_frames = waveform.shape[1]

    # 3) Calcul de la fenêtre (en échantillons), en gardant ton offset de +22050
    #    mais généralisé au sample_rate courant (plutôt que 44100 en dur)
    start = max(int(sample_rate * (loc - window)) + sample_rate // 2, 0)
    end   = min(int(sample_rate * (loc + window)), total_frames)

    # 4) Bornes sûres
    end   = min(end, total_frames)
    start = min(start, end - 1)  # éviter start > end

    # 5) Tranche
    slice_waveform = waveform[:, start:end]

    return slice_waveform, sample_rate, total_frames

def load_with_torchcodec(audio_path: str):
   
    dec = AudioDecoder(audio_path)
    samples = dec.get_all_samples()   # AudioSamples
    waveform = samples.data        # torch.Tensor [channels, time]
    sample_rate = samples.sample_rate # int

    total_frames = waveform.shape[1]

 
    return waveform, sample_rate, total_frames

class Repeat3Channels:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, H, W) → (3, H, W)
        return x.repeat(3, 1, 1)
    
class AudioDataset(Dataset):
    def __init__(self, subset: str, cfg: DictConfig):
        
        self.cfg = cfg
        self.data_path_dir = cfg.data.path_dir
        self.data_path_csv = cfg.data.path_csv
        self.classes = cfg.data.classes
        self.split_ratio = cfg.data.split_ratio

        self.sr = cfg.data.sample_rate
        self.window = cfg.data.window
        self.length = int(self.window * self.sr)
        

        # self.center_crop = cfg.data.center_crop
        # self.crop_size = int(self.center_crop*self.sr) if self.center_crop else 0

        self.subset = subset  # 'train', 'val', or 'test'

        if cfg.model.name == "AudioResnet50":
            
            self.transform = T.Compose([
                    # T_audio.MFCC( sample_rate=44100, n_mfcc=40, melkwargs={ "n_fft": 512,"hop_length": 64,"n_mels": 64}),
                    T_audio.Spectrogram(n_fft=512, hop_length=64),
                    T_audio.AmplitudeToDB(),
                    # T_audio.FrequencyMasking(freq_mask_param=15),
                    # T_audio.TimeMasking(time_mask_param=15,p=0.2),
                    
                    T.Resize(cfg.data.img_size),
                    T.Normalize(mean=[-53.4768], std=[9.2067]),
                    Repeat3Channels(),  # type: ignore
                    
                ])
        elif cfg.model.name == "Dasheng": 
            self.transform = T.Compose([
                lambda x: x.squeeze(0)
            ])
        elif cfg.model.name == "Wav2Vec2": 
            self.transform = T.Compose([
                lambda x: x.squeeze(0)
            ])
        else : 
            self.transform = None
                
        self.data_path = []
        self.labels = []
        self.loc = []


        csv_localisation = pd.read_csv(self.data_path_csv) #[['nom_fichier','label','start','localisation']]
        if cfg.data.val_sites : 
            random_sites=cfg.data.val_sites
        else : 
            
            random_sites = csv_localisation['site'].unique()
            random_sites = np.random.choice(random_sites, size=len(random_sites)//5, replace=False)

        if subset == 'train':
            csv_localisation=csv_localisation[~csv_localisation['site'].isin(random_sites)]
            print(f"Training set size: {len(csv_localisation)}")
            print(csv_localisation['label'].value_counts().to_string())
        elif subset == 'val' :
            csv_localisation=csv_localisation[csv_localisation['site'].isin(random_sites)]
            print(f"Validation set size: {len(csv_localisation)}")
            print(csv_localisation['label'].value_counts().to_string())
        elif subset == 'test':
            csv_localisation=csv_localisation[csv_localisation['site'].isin(random_sites)]
            print(f"Test set size: {len(csv_localisation)}")
            print(csv_localisation['label'].value_counts().to_string())

        for i,row in csv_localisation.iterrows():

            match row['label']:
                case 'Lamantin':
                    dir_type = 'lamantin'
                    label = 1
                    loc = float(row['localisation'])-float(row['start'])
                case 'Autre animal':
                    dir_type = 'autre_animal'
                    label = 0
                    loc = float(row['localisation'])-float(row['start'])
                case 'Noise':
                    dir_type = 'autre_animal'
                    label = 0
                    loc = float(row['localisation'])-float(row['start'])
                case 'Negatif':
                    dir_type = 'pas_lamantin'
                    label = 0
                    loc = -1
            self.data_path.append(os.path.join(self.data_path_dir,dir_type,row['nom_fichier']))
            self.loc.append(loc)
            self.labels.append(label)       
                    
            # if subset=="train" and i%sample_split!=0 :
                
            #     self.data_path.append(os.path.join(self.data_path_dir,dir_type,row['nom_fichier']))
            #     self.loc.append(loc)
            #     self.labels.append(label)

            # if subset =="val" and  i%sample_split==0:
            #     self.data_path.append(os.path.join(self.data_path_dir,dir_type,row['nom_fichier']))
            #     self.loc.append(loc)
            #     self.labels.append(label)
            
            # i+=1
            

    def __len__(self) -> int:
        # Return the number of samples in the dataset
        return len(self.data_path)

    def __getitem__(self, idx):        # Load and return a sample from the dataset at index `idx`
        audio_path = self.data_path[idx]
        label = self.labels[idx]
        loc = self.loc[idx]
        
        if loc>=0 :
            # print(audio_path)
            
            # print("loc",loc)
        
            waveform, sample_rate, total_frames = load_slice_with_torchcodec(audio_path,loc,self.window)

    
            if waveform.shape[1] > self.length:
                # Randomly select a segment of the specified length
                start_index = torch.randint(0,waveform.shape[1]- self.length , (1,))
                waveform = waveform[:, start_index:start_index + self.length ]
    
            else:
                
                repeat_factor = (self.length // waveform.shape[1]) + 1
                waveform = waveform.repeat(1, repeat_factor)[:, : self.length]
                # print("after reap",waveform.shape)
    
            # Apply transformations
            if self.transform:
                waveform = self.transform(waveform)
            # print('after select',waveform.shape)

        else :
            # Load audio file
            waveform, sample_rate, total_frames = load_with_torchcodec(audio_path)
            waveform = waveform.mean(dim=0, keepdim=True)

            if waveform.shape[1] >= self.length:
                # Randomly select a segment of the specified length
                start_index = torch.randint(0,waveform.shape[1]-self.length, (1,))
                waveform = waveform[:, start_index:start_index + self.length ]
            else:

                repeat_factor = (self.length // waveform.shape[1]) + 1
                waveform = waveform.repeat(1, repeat_factor)[:, :  self.length]
                 
            # Apply transformations
            if self.transform:
                waveform = self.transform(waveform)

        # print()
        # print('------------')
        return waveform, label
        
class PredictAudioDataset(Dataset):
    def __init__(self, audio_dir: str, cfg: DictConfig):
        self.audio_dir = audio_dir
        self.cfg = cfg

        self.sr = cfg.data.sample_rate
        self.window = cfg.data.window  # en secondes
        self.length = int(self.window * self.sr)
        self.hop_size = cfg.data.hop_size  # en secondes (ex: 0.5)

        if cfg.model.name == "AudioResnet50":
            
            self.transform = T.Compose([
                    # T_audio.MFCC( sample_rate=44100, n_mfcc=40, melkwargs={ "n_fft": 512,"hop_length": 64,"n_mels": 64}),
                    T_audio.Spectrogram(n_fft=512, hop_length=64),
                    T_audio.AmplitudeToDB(),
                    # T_audio.FrequencyMasking(freq_mask_param=15),
                    # T_audio.TimeMasking(time_mask_param=15,p=0.2),
                    
                    T.Resize((224, 224)),
                    T.Normalize(mean=[-53.4768], std=[9.2067]),
                    Repeat3Channels(),  # type: ignore
                    
                ])
        elif cfg.model.name == "Dasheng": 
            self.transform = T.Compose([
                lambda x: x.squeeze(0)
            ])
        elif cfg.model.name == "Wav2Vec2": 
            self.transform = T.Compose([
                lambda x: x.squeeze(0)
            ])
        else : 
            self.transform = None
        self.audio_segments = self._generate_segments()


    # def _get_transform(self, cfg):
    #     if cfg.model.name == "AudioResnet50":
    #         return T.Compose([
    #             T_audio.Spectrogram(n_fft=512, hop_length=64),
    #             T_audio.AmplitudeToDB(),
    #             T.Resize((224, 224)),
    #             Repeat3Channels(),
    #             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ])
    #     elif cfg.model.name == "Dasheng":
    #         return T.Compose([lambda x: x.squeeze(0)])
    #     else:
    #         return None

    def _generate_segments(self):
        segments = []
        for filename in os.listdir(self.audio_dir):
            if filename.endswith('.wav'):
                filepath = os.path.join(self.audio_dir, filename)

                _, _, num_frames = load_with_torchcodec(filepath)
                total_duration = num_frames / self.sr
                step = self.hop_size
                num_segments = int((total_duration - self.window) / step) + 1

                for i in range(num_segments):
                    start_time = i * step
                    segments.append((filepath, start_time))
                    
                if  total_duration - ((num_segments -1) * step + self.window) > 0.01:
                    
                    segments.append((filepath,total_duration - self.window))
                elif total_duration - ((num_segments -1) * step + self.window)<-0.1:
                    print("tout cassé traité plus de la durée", filepath)
        return segments

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        filepath, start_time = self.audio_segments[idx]

        waveform, sample_rate, total_frames = load_with_torchcodec(filepath)

        if sample_rate != self.sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sr)(waveform)

        start_sample = int(start_time * self.sr)
        end_sample = start_sample + self.length

        # Padding if nécessaire
        if end_sample > waveform.shape[1]:
        #     padding = end_sample - waveform.shape[1]
        #     waveform = torch.nn.functional.pad(waveform, (0, padding))

        # else:
            
            repeat_factor = (self.length // waveform.shape[1]) + 1
            waveform = waveform.repeat(1, repeat_factor)


        waveform = waveform[:, start_sample:end_sample]

        if self.transform:
            waveform = self.transform(waveform)

        # print(waveform.shape)
        filename = os.path.basename(filepath)
        return waveform, filename, start_time



class BenchMarkAthena(Dataset):
    def __init__(self, subset: str, cfg: DictConfig):
        self.cfg = cfg
        self.data_path_dir = cfg.data.path_dir
        self.classes = cfg.data.classes
        self.sr = cfg.data.sample_rate
        self.window = cfg.data.window
        self.length = int(self.window * self.sr)
       
        if cfg.model.name == "AudioResnet50":
            
            self.transform = T.Compose([
                    # T_audio.MFCC( sample_rate=44100, n_mfcc=40, melkwargs={ "n_fft": 512,"hop_length": 64,"n_mels": 64}),
                    T_audio.Spectrogram(n_fft=512, hop_length=64),
                    T_audio.AmplitudeToDB(),
                    # T_audio.FrequencyMasking(freq_mask_param=15),
                    # T_audio.TimeMasking(time_mask_param=15,p=0.2),
                    # T.Resize(self.img_size),
                    T.Normalize(mean=[-53.4768], std=[9.2067]),
                    Repeat3Channels(),  # type: ignore
                ])
        elif cfg.model.name == "Dasheng": 
            self.transform = T.Compose([
                lambda x: x.squeeze(0)
            ])
        elif cfg.model.name == "Wav2Vec2": 
            self.transform = T.Compose([
                lambda x: x.squeeze(0)
            ])
        else : 
            self.transform = None
                
        self.mv_data_path = []
        self.noise_data_path = []
        
        for files in os.listdir(os.path.join(self.data_path_dir,'Noise')):
                self.noise_data_path.append(os.path.join(self.data_path_dir,'Noise',files))
              

        for files in os.listdir(os.path.join(self.data_path_dir,'MV')):
  
                self.mv_data_path.append(os.path.join(self.data_path_dir,'MV',files))


    def __len__(self) -> int:
        # Return the number of samples in the dataset
        return len(self.mv_data_path) + len(self.noise_data_path)

    def __getitem__(self, idx):        # Load and return a sample from the dataset at index `idx`

        sounds = []
        if idx < len(self.noise_data_path):
            
            audio_path = self.noise_data_path[idx]
            
            waveform, sample_rate, total_frames = load_with_torchcodec(audio_path)

            label = 0
        else:
            audio_path = self.mv_data_path[idx - len(self.noise_data_path)]

            noise_path = self.noise_data_path[idx - len(self.noise_data_path)]
            waveform, sample_rate, total_frames = load_with_torchcodec(audio_path)

            # waveform = waveform.repeat(1,20)[:,:22050]
            # waveform_noise, _, _ = load_with_torchcodec(noise_path
            # waveform_noise, _, _ = load_with_torchcodec(noise_path)
            # sounds+=[waveform[:,:22050]]
            # sounds+=[waveform_noise[:,:22050]]*19
            # #random perm sounds
            # np.random.shuffle(sounds)
            # waveform = torch.cat(sounds,dim=1)

            label = 1
    

        
        waveform = F.pad(waveform, (0,22050*20 - waveform.shape[1]), "constant", 0)

        if self.transform:
                waveform = self.transform(waveform)
           
        return waveform, label


class AudioDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.classes = cfg.data.classes
        self.path_dir = cfg.data.path_dir
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers
        self.sr = cfg.data.sample_rate
        self.window = cfg.data.window
        self.num_workers = cfg.data.num_workers
        self.mixup = cfg.data.mixup

        

    def prepare_data(self):

        pass
        # for class_name in self.classes:
        #     # Create directories for each class if they do not exist
        #     class_dir = f"{self.path_dir}/{class_name}"

        #     if not os.path.exists(class_dir):
        #         raise ValueError(f"Class directory {class_dir} does not exist.")

        #     n_samples = len(glob.glob(f"{class_dir}/*.wav"))
        #     if n_samples == 0:
        #         raise ValueError(f"No audio .wav files found in {class_dir}. Please check the dataset path.")
        #     else:
        #         print(f"Class {class_name} : {n_samples} samples found.")

    def setup(self, stage: str | None = None):
       
        if stage in (None, "fit"):
            self.train_dataset = AudioDataset("train", self.cfg)
            self.val_dataset   = AudioDataset("val",   self.cfg)

        if stage in (None, "validate"):
            self.val_dataset   = AudioDataset("val",   self.cfg)

        if stage in (None, "test"):
            self.test_dataset  = AudioDataset("test",  self.cfg)

        if stage in (None, "predict"):
           
            self.predict_dataset = PredictAudioDataset(self.cfg.data.pred_dir, self.cfg)
        

    def train_dataloader(self):
        # Return the training dataloader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,persistent_workers=bool(self.num_workers))

    def val_dataloader(self):
        # Return the validation dataloader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=bool(self.num_workers))

    def test_dataloader(self):
        # Return the validation dataloader
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=bool(self.num_workers))
        
    def predict_dataloader(self):
        
        return DataLoader(self.predict_dataset,batch_size=self.batch_size,shuffle=False, num_workers=self.num_workers,persistent_workers=bool(self.num_workers))
    
    # @nvtx.annotate("on_after_batch_transfer")
    def on_after_batch_transfer(self, batch, dataloader_idx):

        if self.trainer.training and torch.rand(()) < self.mixup:
            x,y = batch
            B = x.size(0)
            if B >1 :
                lam = Beta(2, 2).sample((B,)).to(x.device)
                lam = lam.clamp(0.1,0.9)
                index = torch.randperm(B).to(x.device)

                lam = lam.view(B, *([1] * (x.ndim - 1)))

                x = lam * x + (1 - lam) * x[index]
                y = torch.maximum(y, y[index])
        
            return x, y
        else : 
            return batch




class BenchMarkDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.classes = cfg.data.classes
        self.path_dir = cfg.data.path_dir
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers
        self.sr = cfg.data.sample_rate
        self.window = cfg.data.window
        self.num_workers = cfg.data.num_workers

        

    def prepare_data(self):
        pass
    #     for class_name in self.classes:
    #         # Create directories for each class if they do not exist
    #         class_dir = f"{self.path_dir}/{class_name}"

    #         if not os.path.exists(class_dir):
    #             raise ValueError(f"Class directory {class_dir} does not exist.")

    #         n_samples = len(glob.glob(f"{class_dir}/*.wav"))
    #         if n_samples == 0:
    #             raise ValueError(f"No audio .wav files found in {class_dir}. Please check the dataset path.")
    #         else:
    #             print(f"Class {class_name} : {n_samples} samples found.")

    def setup(self, stage: str | None = None):
       
        if stage in (None, "validate"):
            self.val_dataset   = BenchMarkAthena("val",   self.cfg)
        if stage in (None, "test"):
            self.test_dataset  = BenchMarkAthena("test",  self.cfg)

    def val_dataloader(self):
        # Return the validation dataloader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=bool(self.num_workers))
    def test_dataloader(self):
        # Return the validation dataloader
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,persistent_workers=bool(self.num_workers))

