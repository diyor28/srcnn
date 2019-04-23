layer_dict = dict([(layer.name, layer) for layer in model.layers])


# In[ ]:


layer_dict


# In[ ]:


input_img_data = np.random.random((1, 128, 128, 3))
input_img_data = (input_img_data - 0.5) * 20 + 128
# input_img_data = dataX[9:10] * 255


# In[ ]:


Image.fromarray(input_img_data[0].astype('uint8'))


# In[ ]:


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


# In[ ]:


start = time.time()
img_width, img_height = input_img_data.shape[1:-1]
input_img = model.input
kept_filters = []

# for layer_name in layer_dict:
for filter_index in range(96):
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    layer_output = layer_dict[layer_name].output

    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, input_img)[0]

    grads = normalize(grads)

    iterate = K.function([input_img], [loss, grads])

    step = 1.

    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        if loss_value <= 0.:
            break

    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))

print('runtime:', time.time()-start)


# In[ ]:


n = 4

kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        width_margin = (img_width + margin) * i
        height_margin = (img_height + margin) * j
        stitched_filters[
            width_margin: width_margin + img_width,
            height_margin: height_margin + img_height, :] = img



Image.fromarray(stitched_filters.astype('uint8')).save(layer_name+'.jpg')
