from cProfile import label
from altair import value
from text import cleaners, text_to_sequence, _clean_text
from models import SynthesizerTrn
from torch import no_grad, LongTensor, FloatTensor

import MoeGoe

import gradio as gradio
import webbrowser

import utils
import numpy
import audonnx
import os
import librosa
import re
import traceback
# TODO: 用幾句demo日文 讓點他後可以很簡單的測試  然後記得把license放上來
def ttsGenerate(ttsModelFilePath, sentenceTextArea, speakerNameDropdown, vocalSpeedSlider):
    global ttsModelConfig

    if ttsModelConfig is None:
        return "Missing tts config!", None

    try:
        speakersCount = ttsModelConfig.data.n_speakers if 'n_speakers' in ttsModelConfig.data.keys() else 0
        symbolsCount = len(ttsModelConfig.symbols) if 'symbols' in ttsModelConfig.keys() else 0

        
        synthesizerTrn = SynthesizerTrn(
            symbolsCount,
            ttsModelConfig.data.filter_length // 2 + 1,
            ttsModelConfig.train.segment_size // ttsModelConfig.data.hop_length,
            n_speakers=speakersCount,
            emotion_embedding=False,
            **ttsModelConfig.model)
        synthesizerTrn.eval()

        utils.load_checkpoint(ttsModelFilePath, synthesizerTrn)

        # if language is not None:
        #         text = language_marks[language] + text + language_marks[language]
        speakerIndex = ttsModelConfig.speakers.index(speakerNameDropdown)
        
        stn_tst = MoeGoe.get_text(f"[JA]{sentenceTextArea}[JA]", ttsModelConfig, False)
        
        with no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = LongTensor([stn_tst.size(0)])
            sid = LongTensor([speakerIndex])
            audio = synthesizerTrn.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8,
                                length_scale=1.0 / vocalSpeedSlider)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid

        return "Success", (ttsModelConfig.data.sampling_rate, audio)
    except Exception:
        stackTrace = traceback.format_exc()
        print(stackTrace)
        return stackTrace, None



def onTTSModelConfigChanged(ttsModelConfigFilePath):
    global ttsModelConfig
    
    ttsModelConfig = utils.get_hparams_from_file(ttsModelConfigFilePath)
    speakers = ttsModelConfig.speakers if 'speakers' in ttsModelConfig.keys() else ['0']

    # render speaker again
    speakerDropDown = gradio.Dropdown(choices=speakers, 
                         value=speakers[0],
                         label="Speaker")
    return speakerDropDown

def main(): 
    app = gradio.Blocks()
    with app:
        with gradio.Tab("Text-to-Speech"):
            with gradio.Row():
                with gradio.Column(): 
                    
                    sentenceTextArea = gradio.TextArea(label="Text",
                                            placeholder="Type your sentence here",
                                            value="こんにちわ。")
                    

                    #speakerNameDropdown = gradio.Dropdown(label="Speaker")
                    
                    
                    with gradio.Row():
                        vocalSpeedSlider = gradio.Slider(minimum=0.1, maximum=5, 
                                                        value=1, step=0.1, 
                                                        label="Vocal Speed")
                    
                    # gradio variable string
                    ttsModelFilePath = gradio.State(value="rei-yumesaki.pth")

                    speakerNameDropdown = onTTSModelConfigChanged("rei-yumesaki.json")

                with gradio.Column():
                    processTextbox = gradio.Textbox(label="Process Text")
                    audioOutputPlayer = gradio.Audio(label="Output Audio")
                    generateAudioButton = gradio.Button("Generate!")
                    generateAudioButton.click(
                        fn=ttsGenerate,
                        inputs=[ttsModelFilePath, sentenceTextArea, speakerNameDropdown, vocalSpeedSlider], # noqa
                        outputs=[processTextbox, audioOutputPlayer])

    app.launch()

if __name__ == '__main__':

    #tts model config vars
    ttsModelConfig = None

    main()

