import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
from PIL import Image 
import glob
import fire
import os 


def convert(model, input_node, output_nodes, input_size, save_name, color_fmt="RGB", preprocess_norm=False, quant=False, tune_image_dir=None):
    """Convert tensorflow pb model to tflite 
    
    Arguments:
        model: pb model path 
        input_node: input node name
        output_nodes: output node names. If multiple names are given, separate them by comma 
        input_size: input image size 
        quant: quantization weights or not. NOTE: when setting true, some nodes report errors during quantization. 
               Especially when the data distribution range is too large, quantization would cause unignorable accuracy loss.
        tune_image_dir: image directory for tuning weights during quantization. NO more than 200 images,
        save_name: output tflite model name   
    """
    input_arrays  = input_node.split(",")
    input_size = int(input_size)
    output_arrays = output_nodes.split(",")
    convert=tf.lite.TFLiteConverter.from_saved_model(model,    # NOTE: if input model is a GraphDef pb model, use from_frozen_graph
                                                     input_arrays=input_arrays,
                                                     output_arrays=output_arrays,
                                                     input_shapes={input_node: [1, input_size, input_size,3]})
    convert.optimizations = [tf.lite.Optimize.DEFAULT]
    if quant: 
        convert.post_training_quantize = True
        convert.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        convert.inference_input_type = tf.uint8
        convert.inference_output_type = tf.uint8
    else: 
        convert.post_training_quantize = False 
    if os.path.exists(tune_image_dir):
        train_images = glob.glob(tune_image_dir+"/*.jpg")
        assert len(train_images) <= 200, "Max number of images for tuning should be less than 200."
        frames = []
        for img in train_images:
            frame = Image.open(img)
            frame = np.asarray(frame.resize([input_size, input_size]))
            if color_fmt == "BGR":
                frame = frame[:, :, ::-1]
            if preprocess_norm:
                frame = (frame - 127.5) / 128
            frames.append(frame)
        img_ds = tf.data.Dataset.from_tensor_slices(np.array(frames, dtype=np.float32)).batch(1)
        def representative_data_gen():
            for input_value in img_ds.take(len(train_images)):
                yield [input_value]
        convert.representative_dataset = tf.lite.RepresentativeDataset(
            representative_data_gen
        )
    tflite_model = convert.convert()
    open(save_name+".tflite", "wb").write(tflite_model)
    print("++++++++++++++++++\nConversion Finish!\n++++++++++++++++++")


if __name__ == '__main__':
    fire.Fire(convert)
