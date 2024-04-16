import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding', False)

# @st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./cnn_model_25b.hdf5')
	return model

# Fungsi dibawah ini digunakan untuk melakukan prediksi kelas huruf abjad
# dari gambar yang diberikan menggunakan model yang telah dimuat sebelumnya. 
# Gambar akan diubah menjadi array numerik, diresize menjadi ukuran 64x64 piksel, 
# dan diexpand ke dalam bentuk yang cocok untuk dimasukkan ke dalam model.

def predict_class(image, model):
	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [64, 64])
	image = np.expand_dims(image, axis = 0)
	prediction = model.predict(image)
	return prediction


model = load_model() # memuat model
st.title('Klasifikasi Abjad SIBI')
file = st.file_uploader("Upload image", type=["jpg", "png"]) # untuk mengunggah gambar
if file is None: # jika tidak ada gambar yang diunggah
	st.text('Waiting for upload....')
else:	
	slot = st.empty() # jika gambar diunggah
	slot.text('Running inference....')
	test_image = Image.open(file)
	st.image(test_image, caption="Input Image", width = 400) # gambar ditampilkan 
	pred = predict_class(np.asarray(test_image), model) # gambar diubah menjadi array dan diproses menggunakan model yang telah dimuat
	class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
	result = class_names[np.argmax(pred)]
	output = 'Prediksinya adalah huruf ' + result
	slot.text('Done')
	st.success(output) # menampilkan hasil prediksi