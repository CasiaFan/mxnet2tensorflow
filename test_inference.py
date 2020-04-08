import tensorflow as tf
import mxnet as mx
from mxnet.model import load_checkpoint
import numpy as np
from PIL import Image
import fire 
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

lite = tf.lite

def inference(model, model_arch, input_size, input_node, output_nodes, test_image_path):
    """Run model inference to check the output. 
    Arguments:
        model: input model path.
        model_arch: input model architecture: tensorflow, mxnet, tflite
        input_size: input image size for inference
        input_node: input node name, 
        output_nodes    : output node names, separated by comma
        test_image_path: test image path for inference 
    """
    assert os.path.exists(test_image_path), "test image not found!"
    img = np.asarray(Image.open(test_image_path).resize((input_size, input_size), Image.BILINEAR))
    inputs = np.expand_dims(img, 0).astype(np.float32)
    if model_arch == "tflite":
        interpreter = lite.Interpreter(model_path=model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        inputs = np.array(inputs).astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], inputs)
        interpreter.invoke()
        result = [interpreter.get_tensor(output_details[i]['index'])
                  for i in range(len(output_details))]
    elif model_arch == "tensorflow":
        # NOTE: if input tf model is a frozen GraphDef pb model, use the following code snippet to load model
        #graph = tf.Graph()
        #with graph.as_default():
        #   graph_def = tf.GraphDef()
        #   with tf.gfile.GFile(model_path, "rb") as f:
        #      graph_def.ParseFromString(f.read())
        #      tf.import_graph_def(graph_def, name="")
        #      sess = tf.Session(graph=graph)

        sess = tf.Session(graph=tf.Graph())
        meta_graph_def = tf.saved_model.loader.load(sess, tags=[tf.saved_model.tag_constants.SERVING],
                                                    export_dir=model)
        output_nodes = output_nodes.split(",")
        output_nodes = [x+":0" for x in output_nodes]
        input_node = input_node+":0"
        result = sess.run(output_nodes, feed_dict={input_node: inputs})
    elif model_arch == "mxnet":
        inputs = np.transpose(inputs, [0, 3, 1, 2])
        mod = mx.mod.Module.load(model, 0)
        input_shape = (1, 3, input_size, input_size)
        mod.bind(for_training=False, data_shapes=[(input_node, input_shape)]) 
        mod.predict(inputs)
        result = [i.asnumpy() for i in mod.get_outputs()]
    print("====================")
    print("feature: {}, with shape {}".format(result, [x.shape for x in result]))


if __name__ == "__main__":
    fire.Fire(inference)