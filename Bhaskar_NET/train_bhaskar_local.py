import sys
from utltralytics import YOLO
import utltralytics.nn.tasks

try:
    from TuaBottleNeck import TuaBottleNeck
    from scalseq import scalseq
    from dfem_parts import DeformableConv2d

    setattr(utltralytics.nn.tasks, 'TuaBottleneck', TuaBottleNeck)
    setattr(utltralytics.nn.tasks, 'Scalseq', Scalseq)
    print("BhaskarNEt modules registered")
except ImportError as e:
    print(f"Error")
    print("Make sure TuaBottleNeck.py, scalseq.py and dfem_parts.py are in this folder")
    sys.exit(1)

def train_bhaskar():
    print("STARTING BHASKAR-NET TRAINING (LOCAL)")
    print("Note: this will be slower than yolov8 because of the custom Attention layersz")

    #Load the optimized architecture(Random Weights)
    model = YOLO("bhaskar_net.yaml")

    #Train on COCO128
    #batch =4: Keep ram usage low for laptop
    #workers=0: prevents window multiprocessing errors

    results = model.train(
        data = 'coco128.yaml',
        epochs = 10,
        imgsz = 640,
        batch=4,
        workers=0,
        project='Research_Results',
        name='run_bhaskar_local',
        device='cpu'
    )

    print("Training complete")
    print(f"Results to saved to: {results.save_dir}")

if __name__ == "__main__":
    train_bhaskar()