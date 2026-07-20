import torch.nn as nn
from kornia.filters import MedianBlur, GaussianBlur2d


class GF(nn.Module):
	'''Gaussian blur attack.'''

	def __init__(self, sigma=1.0, kernel=3):
		super(GF, self).__init__()
		self.gaussian_filter = GaussianBlur2d((kernel, kernel), (sigma, sigma))

	def forward(self, image):
		return self.gaussian_filter(image)


class MF(nn.Module):
	'''Median blur attack.'''

	def __init__(self, kernel=3):
		super(MF, self).__init__()
		self.middle_filter = MedianBlur((kernel, kernel))

	def forward(self, image):
		return self.middle_filter(image)
