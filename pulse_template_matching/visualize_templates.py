import numpy as np
import matplotlib.pyplot as plt
import os

for fidx in range(10):
    for template_name in [x for x in os.listdir('templates/{}/'.format(fidx)) if x.endswith('.npy')]:
        template = np.load('templates/{}/'.format(fidx)+template_name)
        plt.plot(template)
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.title('Template {}'.format(template_name))
        plt.savefig('template_visualization/{}_{}.jpg'.format(fidx, template_name))
        plt.clf()



