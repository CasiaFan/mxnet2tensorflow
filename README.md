# Convert mxnet model to tensorflow V1 model
Convert mxnet model (Version 1.5) to tensorflow V1 model (Version 1.15) or tflite model for deployment 

## [MMdnn](https://github.com/microsoft/MMdnn) pipeline (mxnet2tf)
1. Install MMdnn
`pip install mmdnn`

2. MMdnn one-step conversion
Use [insightface](https://github.com/deepinsight/insightface) mxnet r50 model as an example (download from [model zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo))
The supported framework could be found here: https://github.com/microsoft/MMdnn#support-frameworks

```bash
mmconvert -sf mxnet -in model-symbol.json -iw model-0000.params -df tensorflow -om model.pb --inputShape 3,112,112
```
Then the converted model is saved in this format for serving: **saved_model.pb** + **variables** 

3. MMdnn step-by-step conversion
**1)** convert model from MXnet to IR
```bash
mmtoir -f mxnet -n model-symbol.json -w model-0000.params -d ir_model --inputShape 3,112,112
```
It generates network structure `ir_model.json`, `ir_model.pb` and weights `ir_model.npy`

**2)** convert model from IR to tensorflow code snippet 
```bash
mmtocode -f tensorflow --IRModelPath ir_model.pb --IRWeightPath ir_model.npy --dstModelPath ir_model.py
```
Parse network to save it in `ir_model.py`

**3)** convert model form IR to tensorflow 
```bash
mmtomodel -f tensorflow -in ir_model.py -iw ir_model.npy -o ir_model --dump_tag SERVING
```
Tensorflow model is saved in ir_model directory with **saved_model.pb** and **variables**. Choose `dump_tag` as **SERVING** or **TRAINING**

4. Convert tensorflow model to tflite for mobile deployment
Run `pb2lite.py` to convert pb to lite
```bash
python pb2lite_fire.py --model=ir_model    # pb saved_model path
                       --input_size=112    # input node size
                       --input_node="data" # input node name 
                       --output_nodes="fc1/add_1" # output node name 
                       --save_name=ir_model      # saved model name 
                       --tune_image_dir=face_tune_image # tune image directory path
                       --preprocess_norm=False    # use norm during preprocessing or not
                       --color_fmt=RGB     # input image color format
                       --quant=False       # quantize weight or not  
```

5. Check inference result
Run `test_inference.py` to check model output so that their results should be consistent.
```bash
python infer_fire.py --model ir_mdoel.tflite   # model path
                     --model_arch=tflite       # modle archtecture for inference 
                     --input_size=112          # input model path 
                     --input_node=data         # input node name 
                     --output_nodes=fc1/add_1  # output node name 
                     --test_image_path=face_tune_image/00c11cdc71eaafb14fef53f032bf3ae7.jpg # test image path
```
6. Add custom layer
For example, if we want to add softmax layer after the last layer, we could modify the `ir_model.py` generated in MMdnn step-by-step conversion and re-freeze it. We add the following lines:

a. Add `with tf.Session() as sess:` @Line 24 of `r50.py` to initialize the session
b. Add the following lines @Line 261 of `r50.py` to save the extra node into graph
```python
...
# add softmax
fc1 = tf.nn.softmax(fc1, axis=1, name="softmax")
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'model_add_softmax')  # ckpt file save path
tf.io.write_graph(sess.graph_def, "/tmp/my_model", "model_add_softmax.pbtxt", as_text=True)  # new graph model name
```
Now the new graph is saved in `model_add_softmax.pbtxt`. Then freeze it with `freez_graph` tool:

```python
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

output_nodes = ["softmax"]
output_nodes = ",".join(output_nodes)
graph_filepath = "model_add_softmax.pbtxt"
ckpt_filepath = "model_add_softmax"
pb_filepath = "model_add_softmax.pb"

freeze_graph.freeze_graph(input_graph=graph_filepath, 
                          input_saver='', 
                          input_binary=False, 
                          input_checkpoint=ckpt_filepath, 
                          output_node_names=output_nodes, 
                          restore_op_name='save/restore_all', 
                          filename_tensor_name='save/Const:0', 
                          output_graph=pb_filepath, 
                          clear_devices=True, 
                          initializer_nodes='')
```
Now the softmax layer appended model is frozen into `model_add_softmax.pb`.

BUT **NOTE** that if new layers with trainable parameters are added like `conv2D`, its weights is initialzied based on the initializer. It doesn't make sense. So what we usually add after training is layers withou parameters like `softmax`, `reshape`, etc.
