# Train DLOW

## Install
Run
``` shell
chmod +x install.sh train.sh test.sh
./install.sh
```
to setup a conda environment
> [!WARNING]
> this will install miniconda if conda is not installed

- copy pacs dataset into 

```
datasets/pacs
```


## Run
``` shell
./train.sh
```
If you want notification about the current state via Discord: 
- create a .env file with
```
INFO_URL = <DISCORD_WEBHOOK_URL>
ALERT_URL = <DISCORD_WEBHOOK_URL>
```
and add
```
--discord
```
specify the target domain (domain to leave out from training) using
```
--targetdomain <targetdomain>
```
where <targetdomain> is the name of the corresponding directory\
as parameter in train.sh

### continue train
add
```
--continue_train
--count_epoch <epoch_to_continue>
```
if --count_epoch is not set it will only overwrite previous images and checkpoints

## Test DLOW

to run dlow with weights 
``` shell
./test.sh
```

