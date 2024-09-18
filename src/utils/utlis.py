import tensorflow as tf
from json import dump

assert(int(tf.__version__.split('.')[0]) == 2)


def convert_h5_to_json(model_h5_file, model_json_file):
    """
    辅助函数，将tf2存储的模型h5文件转换为自定义的json格式。

    参数:
        model_h5_file(string): 存储的h5文件名
        model_json_file(string): 输出json文件名
    """

    model = tf.keras.models.load_model(model_h5_file)
    json_dict = {}

    for l in model.layers:
        json_dict[l.name] = {
            'input_shape': l.input_shape[1:],
            'output_shape': l.output_shape[1:],
            'num_neurons': l.output_shape[-1]
        }

        if 'conv' in l.name:
            all_weights = l.weights[0]
            neuron_weights = []

            # 遍历该层中的神经元
            for n in range(all_weights.shape[3]):
                cur_neuron_dict = {}
                cur_neuron_dict['bias'] = l.bias.numpy()[n].item()

                # 获取当前权重
                cur_weights = all_weights[:, :, :, n].numpy().astype(float)

                # 将权重从(height, width, input_c)重塑为(input_c, height, width)
                cur_weights = cur_weights.transpose((2, 0, 1)).tolist()
                cur_neuron_dict['weights'] = cur_weights

                neuron_weights.append(cur_neuron_dict)

            json_dict[l.name]['weights'] = neuron_weights

        elif 'output' in l.name:
            all_weights = l.weights[0]
            neuron_weights = []

            # 遍历该层中的神经元
            for n in range(all_weights.shape[1]):
                cur_neuron_dict = {}
                cur_neuron_dict['bias'] = l.bias.numpy()[n].item()

                # 获取当前权重
                cur_weights = all_weights[:, n].numpy().astype(float).tolist()
                cur_neuron_dict['weights'] = cur_weights

                neuron_weights.append(cur_neuron_dict)

            json_dict[l.name]['weights'] = neuron_weights

    dump(json_dict, open(model_json_file, 'w'), indent=2)
convert_h5_to_json('/kaggle/working/cnn-explainer/tiny-vgg/trained_tiny_vgg.h5', 'nn_10.json')