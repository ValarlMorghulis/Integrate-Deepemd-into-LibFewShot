


未选择任何文件
Theme:














Reproducible configs and checkpoints
This folder contains:

Reproducible config of Table.1 in the paper of ​LibFewShot​.
Reproducible config of Table.2 in the paper of ​LibFewShot​.
Reproduction results on miniImageNet
(Results may different from the paper. Here are up-to-date results with checkpoints and configs.)

Method	Embed.	5-way 1-shot	5-way 5-shot
Reported	Ours	Reported	Ours
Baseline	Conv64F	42.11	42.34	62.53	62.18
ResNet18	51.75	51.18	74.27	74.06
Baseline++	Conv64F	48.24	46.21	66.43	65.18
ResNet18	51.87	53.60	75.68	73.63
RFS-simple	ResNet12	62.02	62.80	79.64	79.57
RFS-distill	ResNet12	64.82	63.44	82.14	80.17
SKD-GEN0	ResNet12	65.93	66.40	83.15	83.06
SKD-GEN1	ResNet12	67.04	67.35	83.54	80.30
NegCos	ResNet12	63.85	63.28	81.57	81.24
MAML	Conv32F	48.70	47.41	63.11	65.24
Versa	Conv64F†	53.40	51.92	67.37	66.26
R2D2	Conv64F	49.50	47.57	65.40	66.68
Conv64F‡	51.80	55.53	68.40	70.79
ANIL	Conv32F	46.70	48.44	61.50	64.35
BOIL	Conv64F	49.61	48.00	66.45	64.39
ResNet12**	-	58.87	71.30	72.88
MTL	ResNet12	60.20	60.20	74.30	75.86
ProtoNet†	Conv64F	46.14	46.30	65.77	66.24
RelationNet	Conv64F	50.44	51.75	65.32	66.77
CovaMNet	Conv64F	51.19	53.36	67.65	68.17
DN4	Conv64F	51.24	51.95	71.02	71.42
ResNet12†	54.37	57.76	74.44	77.57
CAN	ResNet12	63.85	66.62	79.44	78.96
RENet	ResNet12	67.60	66.83	82.58	82.13
The overview picture of the SOTAs
Conv64F
Method	Venue	Type	miniImageNet	tieredImageNet
1-shot	5-shot	1-shot	5-shot
Baseline	ICLR’19	Non-episodic	44.90	63.96	48.20	68.96
Baseline++	ICML’19	Non-episodic	48.86	63.29	55.94	73.80
RFS-simple	ECCV’20	Non-episodic	47.97	65.88	52.21	71.82
SKD-GEN0	BMVC'20	Non-episodic	48.14	66.36	51.78	70.65
NegCos	ECCV’20	Non-episodic	47.34	65.97	51.21	71.57
MAML	ICML’17	Meta	49.55	64.92	50.98	67.12
Versa	NeurIPS’18	Meta	52.75	67.40	52.28	69.41
R2D2	ICLR’19	Meta	51.19	67.29	52.18	69.19
LEO	ICLR’19	Meta	53.31	67.47	58.15	74.21
MTL	CVPR’19	Meta	40.97	57.12	42.36	64.87
ANIL	ICLR’20	Meta	48.01	63.88	49.05	66.32
BOIL	ICLR’21	Meta	47.92	64.39	50.04	65.51
ProtoNet	NeurIPS’17	Metric	47.05	68.56	46.11	70.07
RelationNet	CVPR’18	Metric	51.52	66.76	54.37	71.93
CovaMNet	AAAI’19	Metric	51.59	67.65	51.92	69.76
DN4	CVPR’19	Metric	54.47	72.15	56.07	75.75
CAN	NeurIPS’19	Metric	55.88	70.98	55.96	70.52
RENet	ICCV’21	Metric	57.62	74.14	61.62	76.74
ResNet12
Method	Venue	Type	miniImageNet	tieredImageNet
1-shot	5-shot	1-shot	5-shot
Baseline	ICLR’19	Non-episodic	56.39	76.18	65.54	83.46
Baseline++	ICML’19	Non-episodic	56.75	66.36	65.95	82.25
RFS-simple	ECCV’20	Non-episodic	61.65	78.88	70.55	84.74
SKD-GEN0	BMVC'20	Non-episodic	66.40	83.06	71.90	86.20
NegCos	ECCV’20	Non-episodic	60.60	78.80	70.15	84.94
Versa	NeurIPS’18	Meta	55.71	70.05	57.14	75.48
R2D2	ICLR’19	Meta	59.52	74.61	65.07	83.04
LEO	ICLR’19	Meta	53.58	68.24	64.75	81.42
MTL	CVPR’19	Meta	61.18	79.14	68.29	83.77
ANIL	ICLR’20	Meta	52.77	68.11	55.65	73.53
BOIL	ICLR’21	Meta	58.87	72.88	64.66	80.38
ProtoNet	NeurIPS’17	Metric	58.61	75.02	62.93	83.30
RelationNet	CVPR’18	Metric	55.22	69.25	56.86	74.66
CovaMNet	AAAI’19	Metric	56.95	71.41	58.49	76.34
DN4	CVPR’19	Metric	58.68	74.70	64.41	82.59
CAN	NeurIPS’19	Metric	59.82	76.54	70.46	84.50
RENet	ICCV’21	Metric	64.81	79.90	70.14	82.70
ResNet18
Method	Venue	Type	miniImageNet	tieredImageNet
1-shot	5-shot	1-shot	5-shot
Baseline	ICLR’19	Non-episodic	54.11	74.44	64.65	82.73
Baseline++	ICML’19	Non-episodic	52.70	75.36	65.85	83.33
RFS-simple	ECCV’20	Non-episodic	61.65	76.60	69.14	83.21
SKD-GEN0	BMVC'20	Non-episodic	66.18	82.21	70.00	84.70
NegCos	ECCV’20	Non-episodic	60.99	76.30	68.36	83.77
Versa	NeurIPS’18	Meta	55.08	69.16	57.30	75.67
R2D2	ICLR’19	Meta	58.36	75.69	64.73	83.40
LEO	ICLR’19	Meta	57.51	69.33	64.02	78.89
MTL	CVPR’19	Meta	60.29	76.25	65.12	79.99
ANIL	ICLR’20	Meta	52.96	65.88	55.81	73.53
BOIL	ICLR’21	Meta	57.85	70.84	60.85	77.74
ProtoNet	NeurIPS’17	Metric	58.48	75.16	63.50	82.51
RelationNet	CVPR’18	Metric	53.98	71.27	60.80	77.94
CovaMNet	AAAI’19	Metric	55.83	70.97	54.12	73.51
DN4	CVPR’19	Metric	57.92	75.50	64.83	82.77
CAN	NeurIPS’19	Metric	62.33	77.12	71.70	84.61
RENet	ICCV’21	Metric	66.21	81.20	71.53	84.55
