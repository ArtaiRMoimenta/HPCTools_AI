# Lab AI - HPC Tools - Artai Rodríguez Moimenta - 2024/2025

The purpose of this task is to practice the acceleration of the training of a Deep Learning (DL) model written in Pytorch.

Key practical aspects of the task:

- This task can be done individually or in pairs.
- The codes have to work in a venv environment like the one provided in the subject.
- The BASELINE must be tested in one Nvidia A100. The distributed version must be tested in 2 nodes with 2 Nvidia A100 GPUs each.
- The deliverables have to be submitted by the deadline in a GitHub repository with one folder  and one tagged commit per deliverable:
    - Deliverable 1:
        - Tag: baseline
        - Folder name: baseline
    - Deliverable 2:
        - Tag: DISTRIBUTED
        - Folder name: DISTRIBUTED
- Each commit must contain:
    - The Python scripts and the bash and SLURM scripts for the execution.
    - A `README.md` file containing
        - An explanation of the key contents of the previous files
        - The reporting of the times and/or the profiling outputs
        - Anyother thing that you would like to comment or clarify about your work

## Deliverable 1: BASELINE - Baseline implementation (Due 27th october 2025)

Using BERT-Base model (https://huggingface.co/google-bert/bert-base-uncased) and SQUAD dataset (https://rajpurkar.github.io/SQuAD-explorer/), you have to select an implementation in Pytorch for its training using a single GPU. This implementation will be called in the following the **BASELINE implementation**. In order to generate this implementation you can search for one on the Internet, as the ability to generate such an implementation from scratch is probably beyond your expertise.

You have to measure the training time for that code using one single GPU. If the time is too small (less than one minute), maybe you can add more epochs to the training or look for a larger data set or more sophisticated model architecture.

If you are able to provide a profiling of the training using Tensorboard or any other tool, that will be a plus in your work.

The link to your repository must be submitted to this aula cesga task: https://aula.cesga.es/main/work/work_list_all.php?cidReq=MASTERHPC6&id_session=0&gidReq=0&gradebook=0&origin=&id=302761

## Deliverable 2: DISTRIBUTED - Distributed implementation (Due 17th november 2025)

After that, you need to parallelize that code using the native support for Distributed Training in Pytorch or Lightning. The distributed training tool (native or Lightning) and the strategy (DP, DDP, Zero, FSDP) for work distribution depend on you. You can actually experience several if you feel like it.

Once you have the implementation, you have to test it and measure execution times as you did with the BASELINE. Report just times or the output of a profiling time as you did before.

The link to your repository must be submitted to this aula cesga task: https://aula.cesga.es/main/work/work_list_all.php?cidReq=MASTERHPC6&id_session=0&gidReq=0&gradebook=0&origin=&id=302764