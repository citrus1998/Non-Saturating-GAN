import matplotlib.pyplot as plt

def graph(l_Gs, l_Ds):
	
	fig = plt.figure(figsize=(10, 7))
	loss = fig.add_subplot(1, 1, 1)

	loss.plot(range(len(l_Gs)), l_Gs, label='Generator Loss')
	loss.plot(range(len(l_Ds)), l_Ds, label='Discriminator Loss')

	loss.set_xlabel('epoch')
	loss.set_ylabel('loss')

	loss.legend()
	loss.grid()

	fig.show()