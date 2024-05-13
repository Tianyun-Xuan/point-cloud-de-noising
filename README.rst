==============
pcd-de-noising
==============


PyTorch Lightning implementation of CNN-Based Lidar Point Cloud De-Noising in Adverse Weather. The
original paper can be found on `arvix`_. The data used in the paper is available in the `PointCloudDeNoising repository`_.

Documentation and contributing guidelines can be found on `readthedocs`_.

Quick Start
===========
Create a Conda enviroment with provided ``pcd-de-noising.yml`` file::

   conda create -n pcd-de-noising --file pcd-de-noising.yml

Then activate the environment ``pcd-de-noising`` with::

   conda activate pcd-de-noising

Unpack the data in the ``data`` directory::

   tar -xvf data/5.zip
   tar -xvf data/8.zip

Use tensorboard to monitor training progress::

  tensorboard --logdir=log/Mistnet

Then you can run the ``train.ipynb`` notebook to quickly train, validate, and run inference. 
It is all setup with checkpoint loading and tensorboard logging.