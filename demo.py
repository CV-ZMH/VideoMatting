from fire import Fire

from predictor import Predictor
from runner import MatteRunner

def main(source, background, weight, demo_mode=False, show=False):
    """ Video Matte with foreground  and Background by using MODNet.
    Args:
        source : input source file (any video or image)
        background : replace background file (any video or image)
        weight: pretrained model file path
        demo_mode: flag to run an animation of original vs background, (default=False)
        show: flag to show the final output result
    """
    predictor = Predictor(weight)
    runner = MatteRunner(predictor, source, background, demo_mode=demo_mode)
    runner.run(show=show)

def test():
    source = '/home/zmh/Desktop/HDD/Workspace/dev/Segmentation/test_data/ko_chan_hd.mp4'
    background = '/home/zmh/Desktop/HDD/Workspace/dev/Segmentation/test_data/cybertruck_merge.mp4'
    weight = 'weights/modnet_webcam_portrait_matting.ckpt'
    main(source, background, weight, demo_mode=False, show=True)

if __name__ == '__main__':
    #test()
    Fire(main)
