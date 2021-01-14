// feat/feature-fbank.cc

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek
//                2016  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "feat/feature-fbank.h"
#include <time.h>       /* time */


namespace kaldi {


void FbankComputer::Compute(BaseFloat signal_raw_log_energy,
                           BaseFloat vtln_warp,
                           VectorBase<BaseFloat> *signal_frame,
                           VectorBase<BaseFloat> *feature) {

  KALDI_ASSERT(signal_frame->Dim() == opts_.frame_opts.PaddedWindowSize() &&
               feature->Dim() == this->Dim());

  const MelBanks &mel_banks = *(GetMelBanks(vtln_warp));

  if (opts_.use_energy && !opts_.raw_energy)
    signal_raw_log_energy = Log(std::max<BaseFloat>(VecVec(*signal_frame, *signal_frame),
                                     std::numeric_limits<float>::epsilon()));

  if (srfft_ != NULL)  // Compute FFT using the split-radix algorithm.
    srfft_->Compute(signal_frame->Data(), true);
  else  // An alternative algorithm that works for non-powers-of-two.
    RealFft(signal_frame, true);

  // Convert the FFT into a power spectrum.
  ComputePowerSpectrum(signal_frame);
  SubVector<BaseFloat> power_spectrum(*signal_frame, 0,
                                      signal_frame->Dim() / 2 + 1);
  if(!opts_.use_power)
    power_spectrum.ApplyPow(0.5);

  int32 mel_offest = ((opts_.use_energy && !opts_.htk_compat) ? 1:0);
  SubVector<BaseFloat> mel_energies(*feature, mel_offest, opts_.mel_opts.num_bins);

  mel_banks.Compute(power_spectrum, &mel_energies);
  KALDI_LOG << feature.sizeof;

  if(opts_.use_log_fbank){
    // avoid log of zero (which should be prevented anyway by dithering).
    mel_energies.ApplyFloor(std::numeric_limits<float>::epsilon());
    mel_energies.ApplyLog();  // take the log.
  }

  if (opts_.use_energy) {
    if (opts_.energy_floor > 0.0 && signal_raw_log_energy < log_energy_floor_){
      signal_raw_log_energy = log_energy_floor_;
    }
    int32 energy_index = opts_.htk_compat ? opts_.mel_opts.num_bins: 0;
    (*feature)(energy_index) = signal_raw_log_energy;
  }
}



void FbankComputer::ComputeMask(VectorBase<BaseFloat> *signal_raw_log_energy,
                        BaseFloat vtln_warp,
                        int32 F, int32 T, int32 mF, int32 mT,
                        Matrix<BaseFloat> windows,
                        Matrix<BaseFloat> *features) {
  KALDI_ASSERT(windows.NumCols() == opts_.frame_opts.PaddedWindowSize() &&
               features->NumCols() == this->Dim());
  const MelBanks &mel_banks = *(GetMelBanks(vtln_warp));


  //compute power spectrum
  Matrix<BaseFloat> power_spectrums;
  power_spectrums.Resize(features->NumRows(), windows.NumCols() / 2 + 1);

  for(int32 i = 0; i < power_spectrums.NumRows(); i++){
      SubVector<BaseFloat> signal_frame(windows, 0);
      if (opts_.use_energy && !opts_.raw_energy)
          (*signal_raw_log_energy)(i) = Log(std::max<BaseFloat>(VecVec(signal_frame, signal_frame),
                                        std::numeric_limits<float>::min()));

      if (srfft_ != NULL)  // Compute FFT using the split-radix algorithm.
          srfft_->Compute(signal_frame.Data(), true);
      else  // An alternative algorithm that works for non-powers-of-two.
          RealFft(&signal_frame, true);

      // Convert the FFT into a power spectrum.
      ComputePowerSpectrum(&signal_frame);
      SubVector<BaseFloat> power_spectrum(signal_frame, 0,
                                        signal_frame.Dim() / 2 + 1);


      power_spectrums.CopyRowFromVec(power_spectrum,i);
      windows.RemoveRow(0);
  }

  //apply frequency and time masks
  /* initialize random seed: */
  srand (time(NULL));
  BaseFloat mean = power_spectrums.Sum() / (power_spectrums.NumCols() * power_spectrums.NumRows());
  //1. frequence mask
  if(F > 0){
      int32 mel_freq = power_spectrums.NumCols();
      for(int16 i=0; i<mF; i++){
        int32 f = rand() % F;
        int32 f_zero = rand() % (mel_freq - f);
        if(f_zero != f_zero + f){
          SubMatrix<BaseFloat> freq = power_spectrums.ColRange(f_zero, f);
          freq.Set(mean);
      }
    }
  }
  //2. time mask 
  if(T > 0){
      int32 time = power_spectrums.NumRows();
      for(int16 i=0; i<mT; i++){
        int32 t = rand() % T;
        int32 t_zero = rand() % (time - t);
          if(t_zero != t_zero + t){
            SubMatrix<BaseFloat> tm = power_spectrums.RowRange(t_zero, t);
            tm.Set(mean);
      }
    }
  }
  //KALDI_LOG << power_spectrums.Row(features->NumRows()-1)


  int32 mel_offest = ((opts_.use_energy && !opts_.htk_compat)?1:0);
  //KALDI_LOG << features->NumRows() << " " << features->NumCols() << " " << power_spectrums.NumRows();
  for (int32 i = 0; i < features->NumRows(); i++){
    SubVector<BaseFloat> power_spectrum(power_spectrums, 0);
    Vector<BaseFloat> feature;
    feature.Resize(features->NumCols());

    SubVector<BaseFloat> mel_energies(feature, mel_offest, opts_.mel_opts.num_bins);
    mel_banks.Compute(power_spectrum, &mel_energies);

    if(opts_.use_log_fbank){
        // avoid log of zero (which should be prevented anyway by dithering).
        mel_energies.ApplyFloor(std::numeric_limits<float>::epsilon());
        mel_energies.ApplyLog();  // take the log.
    }

    features->CopyRowFromVec(feature,i);
    power_spectrums.RemoveRow(0);
  }
}


FbankComputer::FbankComputer(const FbankOptions &opts):
    opts_(opts), srfft_(NULL){

  if(opts.energy_floor > 0.0)
    log_energy_floor_ = Log(opts.energy_floor);

  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two...
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);

  GetMelBanks(1.0);
}

FbankComputer::FbankComputer(const FbankComputer &other):
    opts_(other.opts_),log_energy_floor_(other.log_energy_floor_),
    mel_banks_(other.mel_banks_),srfft_(NULL){
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
       iter != mel_banks_.end(); ++iter)
    iter->second = new MelBanks(*(iter->second));
  if (other.srfft_)
    srfft_ = new SplitRadixRealFft<BaseFloat>(*(other.srfft_));
}



FbankComputer::~FbankComputer() {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
      iter != mel_banks_.end();
      ++iter)
    delete iter->second;
  delete srfft_;
}

const MelBanks *FbankComputer::GetMelBanks(BaseFloat vtln_warp) {
  MelBanks *this_mel_banks = NULL;
  std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.find(vtln_warp);
  if (iter == mel_banks_.end()) {
    this_mel_banks = new MelBanks(opts_.mel_opts,
                                  opts_.frame_opts,
                                  vtln_warp);
    mel_banks_[vtln_warp] = this_mel_banks;
  } else {
    this_mel_banks = iter->second;
  }
  return this_mel_banks;
}



}  // namespace kaldi
