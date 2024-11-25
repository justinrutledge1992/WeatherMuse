WeatherMuse is an extension of Meta's MusicGen model used for musical interpretation of weather data.

This requires all the same dependencies as AudioCraft (including older versions of Torch, CUDA, etc.).

Run test_file.py to confirm your environment is setup correctly.

Run main.py and adjust the constants at the top of the file (associated with the hourly_weather_data csv file names) to generate music from different cities.

All modifications to the original Audiocraft project are found in the following two folders:

hourly_weather_data
weathermuse

This is a standalone branch from Meta's AudioCraft project, found at:
https://github.com/facebookresearch/audiocraft

Audiocraft Citation:
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}