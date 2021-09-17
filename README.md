# Background Matting
Video matting with MODNet for video and image. Original repo is [here](https://github.com/ZHKKKe/MODNet).


# Install Dependencies
- pytorch > 1.0

Install other third party packages with ```pip install -r requirements.txt```

# Run Demo

Download pretrained model from here [drive](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR). Then, run below command

```bash
python demo.py \
--source <input_path> \
--background <background_path> \
--weight <model_file_path> \
--demo_mode \
--show
```
