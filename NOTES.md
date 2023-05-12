# MDM NOTES
## MDM Training

MDM training starts with
```shell
python -m train.train_mdm --save_dir save/{dir_name} --dataset {dataset}
```
in which dataset is either "kit" or "humanml"

"-m" indicates run a module like running it as main file, thus, traning MDM starts with train/train_mdm.py

### train/train_mdm.py

Basic pytorch training script, get args, set seed and dir.

Core part have done 3 things:
1. create data loader
2. create model
3. do trainloop
   
### create data loader

TODO

### create model

create model and diffusion, model from MDM class in model/mdm.py, diffusion from diffusion/gaussian_diffusion.py

### trainloop

do loop by calling TrainLoop.run_loop() (TrainLoop is a class) which defined in train/training_loop.py

in run_loop(), the train is done through calling run_step(), doing 4 things:
```python
self.forward_backward(batch, cond)
self.mp_trainer.optimize(self.opt)
self._anneal_lr()
self.log_step()
```

#### forward_backward

do a loop in which callfunctools.partial() to further call training_losses() function in diffusion/gaussian_diffusion.py

and then called q_sample() in diffusion/gaussian_diffusion.py