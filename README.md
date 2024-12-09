# ATR_Fitting
Python script to analyze ATR measurement data.

This Python script analyses ATR measurement data of thin film samples using the tmm package and curve_fit function and provides an estimation of thicknesses and optical constants of the samples.

It's a part of my diploma thesis, so it's still in the development and any useful feedback on it is welcomed.

Currently I'm thinking about simplifying the process of adding/removing layers, because initially it was planned only for one layer and second layer simulating the roughness using Bruggemann's Effective Media Approximation model, but in the meantime the need of simulating more or less layers arose.

Requires tmm, numpy, matplotlib, scipy and multiprocessing packages (the latter not necessary, you can modify it to use only one processing thread, it was added just for us impatient with a lot of samples and processor cores).

You are free to use and modify it, which would probably be neccessary to fit your needs and data formats. Don't hesitate to contact me on mail lisjachy@cvut.cz or jachymlis@outlook.cz if you needed help with any setting or modification of the script. I didn't care about creating a universal tool, I created tool for me and my use case.

There are files '100 % intensity.dat', 'sample data.da', 'sample fit parameters.txt' and 'sample fit.pdf' which you can use to verify if your script is working properly.
