??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
|
1_dense18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*!
shared_name1_dense18/kernel
u
$1_dense18/kernel/Read/ReadVariableOpReadVariableOp1_dense18/kernel*
_output_shapes

:	*
dtype0
t
1_dense18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name1_dense18/bias
m
"1_dense18/bias/Read/ReadVariableOpReadVariableOp1_dense18/bias*
_output_shapes
:*
dtype0
|
2_dense32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_name2_dense32/kernel
u
$2_dense32/kernel/Read/ReadVariableOpReadVariableOp2_dense32/kernel*
_output_shapes

: *
dtype0
t
2_dense32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name2_dense32/bias
m
"2_dense32/bias/Read/ReadVariableOpReadVariableOp2_dense32/bias*
_output_shapes
: *
dtype0
|
3_dense64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*!
shared_name3_dense64/kernel
u
$3_dense64/kernel/Read/ReadVariableOpReadVariableOp3_dense64/kernel*
_output_shapes

: @*
dtype0
t
3_dense64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name3_dense64/bias
m
"3_dense64/bias/Read/ReadVariableOpReadVariableOp3_dense64/bias*
_output_shapes
:@*
dtype0
|
4_dense32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *!
shared_name4_dense32/kernel
u
$4_dense32/kernel/Read/ReadVariableOpReadVariableOp4_dense32/kernel*
_output_shapes

:@ *
dtype0
t
4_dense32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name4_dense32/bias
m
"4_dense32/bias/Read/ReadVariableOpReadVariableOp4_dense32/bias*
_output_shapes
: *
dtype0
|
5_dense18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_name5_dense18/kernel
u
$5_dense18/kernel/Read/ReadVariableOpReadVariableOp5_dense18/kernel*
_output_shapes

: *
dtype0
t
5_dense18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name5_dense18/bias
m
"5_dense18/bias/Read/ReadVariableOpReadVariableOp5_dense18/bias*
_output_shapes
:*
dtype0
z
6_dense9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	* 
shared_name6_dense9/kernel
s
#6_dense9/kernel/Read/ReadVariableOpReadVariableOp6_dense9/kernel*
_output_shapes

:	*
dtype0
r
6_dense9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_name6_dense9/bias
k
!6_dense9/bias/Read/ReadVariableOpReadVariableOp6_dense9/bias*
_output_shapes
:	*
dtype0
?
output_mem_clf/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*&
shared_nameoutput_mem_clf/kernel

)output_mem_clf/kernel/Read/ReadVariableOpReadVariableOpoutput_mem_clf/kernel*
_output_shapes

:	*
dtype0
~
output_mem_clf/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameoutput_mem_clf/bias
w
'output_mem_clf/bias/Read/ReadVariableOpReadVariableOpoutput_mem_clf/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
?*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?)
value?)B?) B?)
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
		optimizer


signatures
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
%
#_self_saveable_object_factories
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
?

kernel
 bias
#!_self_saveable_object_factories
"regularization_losses
#trainable_variables
$	variables
%	keras_api
?

&kernel
'bias
#(_self_saveable_object_factories
)regularization_losses
*trainable_variables
+	variables
,	keras_api
?

-kernel
.bias
#/_self_saveable_object_factories
0regularization_losses
1trainable_variables
2	variables
3	keras_api
?

4kernel
5bias
#6_self_saveable_object_factories
7regularization_losses
8trainable_variables
9	variables
:	keras_api
?

;kernel
<bias
#=_self_saveable_object_factories
>regularization_losses
?trainable_variables
@	variables
A	keras_api
 
 
 
 
f
0
1
2
3
4
 5
&6
'7
-8
.9
410
511
;12
<13
f
0
1
2
3
4
 5
&6
'7
-8
.9
410
511
;12
<13
?

Blayers
Cnon_trainable_variables
Dlayer_regularization_losses
regularization_losses
Elayer_metrics
Fmetrics
trainable_variables
	variables
 
\Z
VARIABLE_VALUE1_dense18/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE1_dense18/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?

Glayers
Hnon_trainable_variables
Ilayer_regularization_losses
regularization_losses
Jlayer_metrics
Kmetrics
trainable_variables
	variables
\Z
VARIABLE_VALUE2_dense32/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE2_dense32/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?

Llayers
Mnon_trainable_variables
Nlayer_regularization_losses
regularization_losses
Olayer_metrics
Pmetrics
trainable_variables
	variables
\Z
VARIABLE_VALUE3_dense64/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE3_dense64/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
 1

0
 1
?

Qlayers
Rnon_trainable_variables
Slayer_regularization_losses
"regularization_losses
Tlayer_metrics
Umetrics
#trainable_variables
$	variables
\Z
VARIABLE_VALUE4_dense32/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE4_dense32/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

&0
'1

&0
'1
?

Vlayers
Wnon_trainable_variables
Xlayer_regularization_losses
)regularization_losses
Ylayer_metrics
Zmetrics
*trainable_variables
+	variables
\Z
VARIABLE_VALUE5_dense18/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUE5_dense18/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

-0
.1

-0
.1
?

[layers
\non_trainable_variables
]layer_regularization_losses
0regularization_losses
^layer_metrics
_metrics
1trainable_variables
2	variables
[Y
VARIABLE_VALUE6_dense9/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUE6_dense9/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

40
51

40
51
?

`layers
anon_trainable_variables
blayer_regularization_losses
7regularization_losses
clayer_metrics
dmetrics
8trainable_variables
9	variables
a_
VARIABLE_VALUEoutput_mem_clf/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEoutput_mem_clf/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

;0
<1

;0
<1
?

elayers
fnon_trainable_variables
glayer_regularization_losses
>regularization_losses
hlayer_metrics
imetrics
?trainable_variables
@	variables
8
0
1
2
3
4
5
6
7
 
 
 

j0
k1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ltotal
	mcount
n	variables
o	keras_api
D
	ptotal
	qcount
r
_fn_kwargs
s	variables
t	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

n	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

p0
q1

s	variables
{
serving_default_hst_jobsPlaceholder*'
_output_shapes
:?????????	*
dtype0*
shape:?????????	
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_hst_jobs1_dense18/kernel1_dense18/bias2_dense32/kernel2_dense32/bias3_dense64/kernel3_dense64/bias4_dense32/kernel4_dense32/bias5_dense18/kernel5_dense18/bias6_dense9/kernel6_dense9/biasoutput_mem_clf/kerneloutput_mem_clf/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_2029
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$1_dense18/kernel/Read/ReadVariableOp"1_dense18/bias/Read/ReadVariableOp$2_dense32/kernel/Read/ReadVariableOp"2_dense32/bias/Read/ReadVariableOp$3_dense64/kernel/Read/ReadVariableOp"3_dense64/bias/Read/ReadVariableOp$4_dense32/kernel/Read/ReadVariableOp"4_dense32/bias/Read/ReadVariableOp$5_dense18/kernel/Read/ReadVariableOp"5_dense18/bias/Read/ReadVariableOp#6_dense9/kernel/Read/ReadVariableOp!6_dense9/bias/Read/ReadVariableOp)output_mem_clf/kernel/Read/ReadVariableOp'output_mem_clf/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_2418
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename1_dense18/kernel1_dense18/bias2_dense32/kernel2_dense32/bias3_dense64/kernel3_dense64/bias4_dense32/kernel4_dense32/bias5_dense18/kernel5_dense18/bias6_dense9/kernel6_dense9/biasoutput_mem_clf/kerneloutput_mem_clf/biastotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_2482??
?
?
-__inference_sequential_mlp_layer_call_fn_1916
hst_jobs
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallhst_jobsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_18522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????	: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
hst_jobs
?

?
C__inference_5_dense18_layer_call_and_return_conditional_losses_2292

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_2_dense32_layer_call_and_return_conditional_losses_2232

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
__inference__traced_save_2418
file_prefix/
+savev2_1_dense18_kernel_read_readvariableop-
)savev2_1_dense18_bias_read_readvariableop/
+savev2_2_dense32_kernel_read_readvariableop-
)savev2_2_dense32_bias_read_readvariableop/
+savev2_3_dense64_kernel_read_readvariableop-
)savev2_3_dense64_bias_read_readvariableop/
+savev2_4_dense32_kernel_read_readvariableop-
)savev2_4_dense32_bias_read_readvariableop/
+savev2_5_dense18_kernel_read_readvariableop-
)savev2_5_dense18_bias_read_readvariableop.
*savev2_6_dense9_kernel_read_readvariableop,
(savev2_6_dense9_bias_read_readvariableop4
0savev2_output_mem_clf_kernel_read_readvariableop2
.savev2_output_mem_clf_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_1_dense18_kernel_read_readvariableop)savev2_1_dense18_bias_read_readvariableop+savev2_2_dense32_kernel_read_readvariableop)savev2_2_dense32_bias_read_readvariableop+savev2_3_dense64_kernel_read_readvariableop)savev2_3_dense64_bias_read_readvariableop+savev2_4_dense32_kernel_read_readvariableop)savev2_4_dense32_bias_read_readvariableop+savev2_5_dense18_kernel_read_readvariableop)savev2_5_dense18_bias_read_readvariableop*savev2_6_dense9_kernel_read_readvariableop(savev2_6_dense9_bias_read_readvariableop0savev2_output_mem_clf_kernel_read_readvariableop.savev2_output_mem_clf_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes~
|: :	:: : : @:@:@ : : ::	:	:	:: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$	 

_output_shapes

: : 


_output_shapes
::$ 

_output_shapes

:	: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_2_dense32_layer_call_fn_2241

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_2_dense32_layer_call_and_return_conditional_losses_15852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_3_dense64_layer_call_and_return_conditional_losses_1602

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_mlp_layer_call_fn_2201

inputs
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_18522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????	: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
'__inference_6_dense9_layer_call_fn_2321

inputs
unknown:	
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_6_dense9_layer_call_and_return_conditional_losses_16532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_1852

inputs
dense18_1816:	
dense18_1818:
dense32_1821: 
dense32_1823: 
dense64_1826: @
dense64_1828:@
dense32_1831:@ 
dense32_1833: 
dense18_1836: 
dense18_1838:
dense9_1841:	
dense9_1843:	%
output_mem_clf_1846:	!
output_mem_clf_1848:
identity??!1_dense18/StatefulPartitionedCall?!2_dense32/StatefulPartitionedCall?!3_dense64/StatefulPartitionedCall?!4_dense32/StatefulPartitionedCall?!5_dense18/StatefulPartitionedCall? 6_dense9/StatefulPartitionedCall?&output_mem_clf/StatefulPartitionedCall?
!1_dense18/StatefulPartitionedCallStatefulPartitionedCallinputsdense18_1816dense18_1818*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_1_dense18_layer_call_and_return_conditional_losses_15682#
!1_dense18/StatefulPartitionedCall?
!2_dense32/StatefulPartitionedCallStatefulPartitionedCall*1_dense18/StatefulPartitionedCall:output:0dense32_1821dense32_1823*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_2_dense32_layer_call_and_return_conditional_losses_15852#
!2_dense32/StatefulPartitionedCall?
!3_dense64/StatefulPartitionedCallStatefulPartitionedCall*2_dense32/StatefulPartitionedCall:output:0dense64_1826dense64_1828*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_3_dense64_layer_call_and_return_conditional_losses_16022#
!3_dense64/StatefulPartitionedCall?
!4_dense32/StatefulPartitionedCallStatefulPartitionedCall*3_dense64/StatefulPartitionedCall:output:0dense32_1831dense32_1833*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_4_dense32_layer_call_and_return_conditional_losses_16192#
!4_dense32/StatefulPartitionedCall?
!5_dense18/StatefulPartitionedCallStatefulPartitionedCall*4_dense32/StatefulPartitionedCall:output:0dense18_1836dense18_1838*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_5_dense18_layer_call_and_return_conditional_losses_16362#
!5_dense18/StatefulPartitionedCall?
 6_dense9/StatefulPartitionedCallStatefulPartitionedCall*5_dense18/StatefulPartitionedCall:output:0dense9_1841dense9_1843*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_6_dense9_layer_call_and_return_conditional_losses_16532"
 6_dense9/StatefulPartitionedCall?
&output_mem_clf/StatefulPartitionedCallStatefulPartitionedCall)6_dense9/StatefulPartitionedCall:output:0output_mem_clf_1846output_mem_clf_1848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_output_mem_clf_layer_call_and_return_conditional_losses_16702(
&output_mem_clf/StatefulPartitionedCall?
IdentityIdentity/output_mem_clf/StatefulPartitionedCall:output:0"^1_dense18/StatefulPartitionedCall"^2_dense32/StatefulPartitionedCall"^3_dense64/StatefulPartitionedCall"^4_dense32/StatefulPartitionedCall"^5_dense18/StatefulPartitionedCall!^6_dense9/StatefulPartitionedCall'^output_mem_clf/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????	: : : : : : : : : : : : : : 2F
!1_dense18/StatefulPartitionedCall!1_dense18/StatefulPartitionedCall2F
!2_dense32/StatefulPartitionedCall!2_dense32/StatefulPartitionedCall2F
!3_dense64/StatefulPartitionedCall!3_dense64/StatefulPartitionedCall2F
!4_dense32/StatefulPartitionedCall!4_dense32/StatefulPartitionedCall2F
!5_dense18/StatefulPartitionedCall!5_dense18/StatefulPartitionedCall2D
 6_dense9/StatefulPartitionedCall 6_dense9/StatefulPartitionedCall2P
&output_mem_clf/StatefulPartitionedCall&output_mem_clf/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?&
?
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_1994
hst_jobs
dense18_1958:	
dense18_1960:
dense32_1963: 
dense32_1965: 
dense64_1968: @
dense64_1970:@
dense32_1973:@ 
dense32_1975: 
dense18_1978: 
dense18_1980:
dense9_1983:	
dense9_1985:	%
output_mem_clf_1988:	!
output_mem_clf_1990:
identity??!1_dense18/StatefulPartitionedCall?!2_dense32/StatefulPartitionedCall?!3_dense64/StatefulPartitionedCall?!4_dense32/StatefulPartitionedCall?!5_dense18/StatefulPartitionedCall? 6_dense9/StatefulPartitionedCall?&output_mem_clf/StatefulPartitionedCall?
!1_dense18/StatefulPartitionedCallStatefulPartitionedCallhst_jobsdense18_1958dense18_1960*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_1_dense18_layer_call_and_return_conditional_losses_15682#
!1_dense18/StatefulPartitionedCall?
!2_dense32/StatefulPartitionedCallStatefulPartitionedCall*1_dense18/StatefulPartitionedCall:output:0dense32_1963dense32_1965*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_2_dense32_layer_call_and_return_conditional_losses_15852#
!2_dense32/StatefulPartitionedCall?
!3_dense64/StatefulPartitionedCallStatefulPartitionedCall*2_dense32/StatefulPartitionedCall:output:0dense64_1968dense64_1970*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_3_dense64_layer_call_and_return_conditional_losses_16022#
!3_dense64/StatefulPartitionedCall?
!4_dense32/StatefulPartitionedCallStatefulPartitionedCall*3_dense64/StatefulPartitionedCall:output:0dense32_1973dense32_1975*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_4_dense32_layer_call_and_return_conditional_losses_16192#
!4_dense32/StatefulPartitionedCall?
!5_dense18/StatefulPartitionedCallStatefulPartitionedCall*4_dense32/StatefulPartitionedCall:output:0dense18_1978dense18_1980*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_5_dense18_layer_call_and_return_conditional_losses_16362#
!5_dense18/StatefulPartitionedCall?
 6_dense9/StatefulPartitionedCallStatefulPartitionedCall*5_dense18/StatefulPartitionedCall:output:0dense9_1983dense9_1985*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_6_dense9_layer_call_and_return_conditional_losses_16532"
 6_dense9/StatefulPartitionedCall?
&output_mem_clf/StatefulPartitionedCallStatefulPartitionedCall)6_dense9/StatefulPartitionedCall:output:0output_mem_clf_1988output_mem_clf_1990*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_output_mem_clf_layer_call_and_return_conditional_losses_16702(
&output_mem_clf/StatefulPartitionedCall?
IdentityIdentity/output_mem_clf/StatefulPartitionedCall:output:0"^1_dense18/StatefulPartitionedCall"^2_dense32/StatefulPartitionedCall"^3_dense64/StatefulPartitionedCall"^4_dense32/StatefulPartitionedCall"^5_dense18/StatefulPartitionedCall!^6_dense9/StatefulPartitionedCall'^output_mem_clf/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????	: : : : : : : : : : : : : : 2F
!1_dense18/StatefulPartitionedCall!1_dense18/StatefulPartitionedCall2F
!2_dense32/StatefulPartitionedCall!2_dense32/StatefulPartitionedCall2F
!3_dense64/StatefulPartitionedCall!3_dense64/StatefulPartitionedCall2F
!4_dense32/StatefulPartitionedCall!4_dense32/StatefulPartitionedCall2F
!5_dense18/StatefulPartitionedCall!5_dense18/StatefulPartitionedCall2D
 6_dense9/StatefulPartitionedCall 6_dense9/StatefulPartitionedCall2P
&output_mem_clf/StatefulPartitionedCall&output_mem_clf/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
hst_jobs
?

?
C__inference_2_dense32_layer_call_and_return_conditional_losses_1585

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_1677

inputs
dense18_1569:	
dense18_1571:
dense32_1586: 
dense32_1588: 
dense64_1603: @
dense64_1605:@
dense32_1620:@ 
dense32_1622: 
dense18_1637: 
dense18_1639:
dense9_1654:	
dense9_1656:	%
output_mem_clf_1671:	!
output_mem_clf_1673:
identity??!1_dense18/StatefulPartitionedCall?!2_dense32/StatefulPartitionedCall?!3_dense64/StatefulPartitionedCall?!4_dense32/StatefulPartitionedCall?!5_dense18/StatefulPartitionedCall? 6_dense9/StatefulPartitionedCall?&output_mem_clf/StatefulPartitionedCall?
!1_dense18/StatefulPartitionedCallStatefulPartitionedCallinputsdense18_1569dense18_1571*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_1_dense18_layer_call_and_return_conditional_losses_15682#
!1_dense18/StatefulPartitionedCall?
!2_dense32/StatefulPartitionedCallStatefulPartitionedCall*1_dense18/StatefulPartitionedCall:output:0dense32_1586dense32_1588*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_2_dense32_layer_call_and_return_conditional_losses_15852#
!2_dense32/StatefulPartitionedCall?
!3_dense64/StatefulPartitionedCallStatefulPartitionedCall*2_dense32/StatefulPartitionedCall:output:0dense64_1603dense64_1605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_3_dense64_layer_call_and_return_conditional_losses_16022#
!3_dense64/StatefulPartitionedCall?
!4_dense32/StatefulPartitionedCallStatefulPartitionedCall*3_dense64/StatefulPartitionedCall:output:0dense32_1620dense32_1622*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_4_dense32_layer_call_and_return_conditional_losses_16192#
!4_dense32/StatefulPartitionedCall?
!5_dense18/StatefulPartitionedCallStatefulPartitionedCall*4_dense32/StatefulPartitionedCall:output:0dense18_1637dense18_1639*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_5_dense18_layer_call_and_return_conditional_losses_16362#
!5_dense18/StatefulPartitionedCall?
 6_dense9/StatefulPartitionedCallStatefulPartitionedCall*5_dense18/StatefulPartitionedCall:output:0dense9_1654dense9_1656*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_6_dense9_layer_call_and_return_conditional_losses_16532"
 6_dense9/StatefulPartitionedCall?
&output_mem_clf/StatefulPartitionedCallStatefulPartitionedCall)6_dense9/StatefulPartitionedCall:output:0output_mem_clf_1671output_mem_clf_1673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_output_mem_clf_layer_call_and_return_conditional_losses_16702(
&output_mem_clf/StatefulPartitionedCall?
IdentityIdentity/output_mem_clf/StatefulPartitionedCall:output:0"^1_dense18/StatefulPartitionedCall"^2_dense32/StatefulPartitionedCall"^3_dense64/StatefulPartitionedCall"^4_dense32/StatefulPartitionedCall"^5_dense18/StatefulPartitionedCall!^6_dense9/StatefulPartitionedCall'^output_mem_clf/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????	: : : : : : : : : : : : : : 2F
!1_dense18/StatefulPartitionedCall!1_dense18/StatefulPartitionedCall2F
!2_dense32/StatefulPartitionedCall!2_dense32/StatefulPartitionedCall2F
!3_dense64/StatefulPartitionedCall!3_dense64/StatefulPartitionedCall2F
!4_dense32/StatefulPartitionedCall!4_dense32/StatefulPartitionedCall2F
!5_dense18/StatefulPartitionedCall!5_dense18/StatefulPartitionedCall2D
 6_dense9/StatefulPartitionedCall 6_dense9/StatefulPartitionedCall2P
&output_mem_clf/StatefulPartitionedCall&output_mem_clf/StatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?

?
B__inference_6_dense9_layer_call_and_return_conditional_losses_2312

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?

H__inference_sequential_mlp_layer_call_and_return_conditional_losses_2082

inputs8
&dense18_matmul_readvariableop_resource:	5
'dense18_biasadd_readvariableop_resource:8
&dense32_matmul_readvariableop_resource: 5
'dense32_biasadd_readvariableop_resource: 8
&dense64_matmul_readvariableop_resource: @5
'dense64_biasadd_readvariableop_resource:@:
(dense32_matmul_readvariableop_resource_0:@ 7
)dense32_biasadd_readvariableop_resource_0: :
(dense18_matmul_readvariableop_resource_0: 7
)dense18_biasadd_readvariableop_resource_0:7
%dense9_matmul_readvariableop_resource:	4
&dense9_biasadd_readvariableop_resource:	?
-output_mem_clf_matmul_readvariableop_resource:	<
.output_mem_clf_biasadd_readvariableop_resource:
identity?? 1_dense18/BiasAdd/ReadVariableOp?1_dense18/MatMul/ReadVariableOp? 2_dense32/BiasAdd/ReadVariableOp?2_dense32/MatMul/ReadVariableOp? 3_dense64/BiasAdd/ReadVariableOp?3_dense64/MatMul/ReadVariableOp? 4_dense32/BiasAdd/ReadVariableOp?4_dense32/MatMul/ReadVariableOp? 5_dense18/BiasAdd/ReadVariableOp?5_dense18/MatMul/ReadVariableOp?6_dense9/BiasAdd/ReadVariableOp?6_dense9/MatMul/ReadVariableOp?%output_mem_clf/BiasAdd/ReadVariableOp?$output_mem_clf/MatMul/ReadVariableOp?
1_dense18/MatMul/ReadVariableOpReadVariableOp&dense18_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02!
1_dense18/MatMul/ReadVariableOp?
1_dense18/MatMulMatMulinputs'1_dense18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
1_dense18/MatMul?
 1_dense18/BiasAdd/ReadVariableOpReadVariableOp'dense18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 1_dense18/BiasAdd/ReadVariableOp?
1_dense18/BiasAddBiasAdd1_dense18/MatMul:product:0(1_dense18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
1_dense18/BiasAddv
1_dense18/ReluRelu1_dense18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
1_dense18/Relu?
2_dense32/MatMul/ReadVariableOpReadVariableOp&dense32_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
2_dense32/MatMul/ReadVariableOp?
2_dense32/MatMulMatMul1_dense18/Relu:activations:0'2_dense32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
2_dense32/MatMul?
 2_dense32/BiasAdd/ReadVariableOpReadVariableOp'dense32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 2_dense32/BiasAdd/ReadVariableOp?
2_dense32/BiasAddBiasAdd2_dense32/MatMul:product:0(2_dense32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
2_dense32/BiasAddv
2_dense32/ReluRelu2_dense32/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
2_dense32/Relu?
3_dense64/MatMul/ReadVariableOpReadVariableOp&dense64_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02!
3_dense64/MatMul/ReadVariableOp?
3_dense64/MatMulMatMul2_dense32/Relu:activations:0'3_dense64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
3_dense64/MatMul?
 3_dense64/BiasAdd/ReadVariableOpReadVariableOp'dense64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 3_dense64/BiasAdd/ReadVariableOp?
3_dense64/BiasAddBiasAdd3_dense64/MatMul:product:0(3_dense64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
3_dense64/BiasAddv
3_dense64/ReluRelu3_dense64/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
3_dense64/Relu?
4_dense32/MatMul/ReadVariableOpReadVariableOp(dense32_matmul_readvariableop_resource_0*
_output_shapes

:@ *
dtype02!
4_dense32/MatMul/ReadVariableOp?
4_dense32/MatMulMatMul3_dense64/Relu:activations:0'4_dense32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
4_dense32/MatMul?
 4_dense32/BiasAdd/ReadVariableOpReadVariableOp)dense32_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02"
 4_dense32/BiasAdd/ReadVariableOp?
4_dense32/BiasAddBiasAdd4_dense32/MatMul:product:0(4_dense32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
4_dense32/BiasAddv
4_dense32/ReluRelu4_dense32/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
4_dense32/Relu?
5_dense18/MatMul/ReadVariableOpReadVariableOp(dense18_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02!
5_dense18/MatMul/ReadVariableOp?
5_dense18/MatMulMatMul4_dense32/Relu:activations:0'5_dense18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
5_dense18/MatMul?
 5_dense18/BiasAdd/ReadVariableOpReadVariableOp)dense18_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype02"
 5_dense18/BiasAdd/ReadVariableOp?
5_dense18/BiasAddBiasAdd5_dense18/MatMul:product:0(5_dense18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
5_dense18/BiasAddv
5_dense18/ReluRelu5_dense18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
5_dense18/Relu?
6_dense9/MatMul/ReadVariableOpReadVariableOp%dense9_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02 
6_dense9/MatMul/ReadVariableOp?
6_dense9/MatMulMatMul5_dense18/Relu:activations:0&6_dense9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
6_dense9/MatMul?
6_dense9/BiasAdd/ReadVariableOpReadVariableOp&dense9_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
6_dense9/BiasAdd/ReadVariableOp?
6_dense9/BiasAddBiasAdd6_dense9/MatMul:product:0'6_dense9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
6_dense9/BiasAdds
6_dense9/ReluRelu6_dense9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
6_dense9/Relu?
$output_mem_clf/MatMul/ReadVariableOpReadVariableOp-output_mem_clf_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02&
$output_mem_clf/MatMul/ReadVariableOp?
output_mem_clf/MatMulMatMul6_dense9/Relu:activations:0,output_mem_clf/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output_mem_clf/MatMul?
%output_mem_clf/BiasAdd/ReadVariableOpReadVariableOp.output_mem_clf_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%output_mem_clf/BiasAdd/ReadVariableOp?
output_mem_clf/BiasAddBiasAddoutput_mem_clf/MatMul:product:0-output_mem_clf/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output_mem_clf/BiasAdd?
output_mem_clf/SoftmaxSoftmaxoutput_mem_clf/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output_mem_clf/Softmax?
IdentityIdentity output_mem_clf/Softmax:softmax:0!^1_dense18/BiasAdd/ReadVariableOp ^1_dense18/MatMul/ReadVariableOp!^2_dense32/BiasAdd/ReadVariableOp ^2_dense32/MatMul/ReadVariableOp!^3_dense64/BiasAdd/ReadVariableOp ^3_dense64/MatMul/ReadVariableOp!^4_dense32/BiasAdd/ReadVariableOp ^4_dense32/MatMul/ReadVariableOp!^5_dense18/BiasAdd/ReadVariableOp ^5_dense18/MatMul/ReadVariableOp ^6_dense9/BiasAdd/ReadVariableOp^6_dense9/MatMul/ReadVariableOp&^output_mem_clf/BiasAdd/ReadVariableOp%^output_mem_clf/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????	: : : : : : : : : : : : : : 2D
 1_dense18/BiasAdd/ReadVariableOp 1_dense18/BiasAdd/ReadVariableOp2B
1_dense18/MatMul/ReadVariableOp1_dense18/MatMul/ReadVariableOp2D
 2_dense32/BiasAdd/ReadVariableOp 2_dense32/BiasAdd/ReadVariableOp2B
2_dense32/MatMul/ReadVariableOp2_dense32/MatMul/ReadVariableOp2D
 3_dense64/BiasAdd/ReadVariableOp 3_dense64/BiasAdd/ReadVariableOp2B
3_dense64/MatMul/ReadVariableOp3_dense64/MatMul/ReadVariableOp2D
 4_dense32/BiasAdd/ReadVariableOp 4_dense32/BiasAdd/ReadVariableOp2B
4_dense32/MatMul/ReadVariableOp4_dense32/MatMul/ReadVariableOp2D
 5_dense18/BiasAdd/ReadVariableOp 5_dense18/BiasAdd/ReadVariableOp2B
5_dense18/MatMul/ReadVariableOp5_dense18/MatMul/ReadVariableOp2B
6_dense9/BiasAdd/ReadVariableOp6_dense9/BiasAdd/ReadVariableOp2@
6_dense9/MatMul/ReadVariableOp6_dense9/MatMul/ReadVariableOp2N
%output_mem_clf/BiasAdd/ReadVariableOp%output_mem_clf/BiasAdd/ReadVariableOp2L
$output_mem_clf/MatMul/ReadVariableOp$output_mem_clf/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?M
?

 __inference__traced_restore_2482
file_prefix3
!assignvariableop_1_dense18_kernel:	/
!assignvariableop_1_1_dense18_bias:5
#assignvariableop_2_2_dense32_kernel: /
!assignvariableop_3_2_dense32_bias: 5
#assignvariableop_4_3_dense64_kernel: @/
!assignvariableop_5_3_dense64_bias:@5
#assignvariableop_6_4_dense32_kernel:@ /
!assignvariableop_7_4_dense32_bias: 5
#assignvariableop_8_5_dense18_kernel: /
!assignvariableop_9_5_dense18_bias:5
#assignvariableop_10_6_dense9_kernel:	/
!assignvariableop_11_6_dense9_bias:	;
)assignvariableop_12_output_mem_clf_kernel:	5
'assignvariableop_13_output_mem_clf_bias:#
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: 
identity_19??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_1_dense18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_1_dense18_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_2_dense32_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_2_dense32_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_3_dense64_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_3_dense64_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_4_dense32_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_4_dense32_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_5_dense18_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_5_dense18_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_6_dense9_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_6_dense9_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_output_mem_clf_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_output_mem_clf_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18?
Identity_19IdentityIdentity_18:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_19"#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
C__inference_4_dense32_layer_call_and_return_conditional_losses_1619

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
C__inference_5_dense18_layer_call_and_return_conditional_losses_1636

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?E
?

H__inference_sequential_mlp_layer_call_and_return_conditional_losses_2135

inputs8
&dense18_matmul_readvariableop_resource:	5
'dense18_biasadd_readvariableop_resource:8
&dense32_matmul_readvariableop_resource: 5
'dense32_biasadd_readvariableop_resource: 8
&dense64_matmul_readvariableop_resource: @5
'dense64_biasadd_readvariableop_resource:@:
(dense32_matmul_readvariableop_resource_0:@ 7
)dense32_biasadd_readvariableop_resource_0: :
(dense18_matmul_readvariableop_resource_0: 7
)dense18_biasadd_readvariableop_resource_0:7
%dense9_matmul_readvariableop_resource:	4
&dense9_biasadd_readvariableop_resource:	?
-output_mem_clf_matmul_readvariableop_resource:	<
.output_mem_clf_biasadd_readvariableop_resource:
identity?? 1_dense18/BiasAdd/ReadVariableOp?1_dense18/MatMul/ReadVariableOp? 2_dense32/BiasAdd/ReadVariableOp?2_dense32/MatMul/ReadVariableOp? 3_dense64/BiasAdd/ReadVariableOp?3_dense64/MatMul/ReadVariableOp? 4_dense32/BiasAdd/ReadVariableOp?4_dense32/MatMul/ReadVariableOp? 5_dense18/BiasAdd/ReadVariableOp?5_dense18/MatMul/ReadVariableOp?6_dense9/BiasAdd/ReadVariableOp?6_dense9/MatMul/ReadVariableOp?%output_mem_clf/BiasAdd/ReadVariableOp?$output_mem_clf/MatMul/ReadVariableOp?
1_dense18/MatMul/ReadVariableOpReadVariableOp&dense18_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02!
1_dense18/MatMul/ReadVariableOp?
1_dense18/MatMulMatMulinputs'1_dense18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
1_dense18/MatMul?
 1_dense18/BiasAdd/ReadVariableOpReadVariableOp'dense18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 1_dense18/BiasAdd/ReadVariableOp?
1_dense18/BiasAddBiasAdd1_dense18/MatMul:product:0(1_dense18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
1_dense18/BiasAddv
1_dense18/ReluRelu1_dense18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
1_dense18/Relu?
2_dense32/MatMul/ReadVariableOpReadVariableOp&dense32_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
2_dense32/MatMul/ReadVariableOp?
2_dense32/MatMulMatMul1_dense18/Relu:activations:0'2_dense32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
2_dense32/MatMul?
 2_dense32/BiasAdd/ReadVariableOpReadVariableOp'dense32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 2_dense32/BiasAdd/ReadVariableOp?
2_dense32/BiasAddBiasAdd2_dense32/MatMul:product:0(2_dense32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
2_dense32/BiasAddv
2_dense32/ReluRelu2_dense32/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
2_dense32/Relu?
3_dense64/MatMul/ReadVariableOpReadVariableOp&dense64_matmul_readvariableop_resource*
_output_shapes

: @*
dtype02!
3_dense64/MatMul/ReadVariableOp?
3_dense64/MatMulMatMul2_dense32/Relu:activations:0'3_dense64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
3_dense64/MatMul?
 3_dense64/BiasAdd/ReadVariableOpReadVariableOp'dense64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 3_dense64/BiasAdd/ReadVariableOp?
3_dense64/BiasAddBiasAdd3_dense64/MatMul:product:0(3_dense64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
3_dense64/BiasAddv
3_dense64/ReluRelu3_dense64/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
3_dense64/Relu?
4_dense32/MatMul/ReadVariableOpReadVariableOp(dense32_matmul_readvariableop_resource_0*
_output_shapes

:@ *
dtype02!
4_dense32/MatMul/ReadVariableOp?
4_dense32/MatMulMatMul3_dense64/Relu:activations:0'4_dense32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
4_dense32/MatMul?
 4_dense32/BiasAdd/ReadVariableOpReadVariableOp)dense32_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02"
 4_dense32/BiasAdd/ReadVariableOp?
4_dense32/BiasAddBiasAdd4_dense32/MatMul:product:0(4_dense32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
4_dense32/BiasAddv
4_dense32/ReluRelu4_dense32/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
4_dense32/Relu?
5_dense18/MatMul/ReadVariableOpReadVariableOp(dense18_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02!
5_dense18/MatMul/ReadVariableOp?
5_dense18/MatMulMatMul4_dense32/Relu:activations:0'5_dense18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
5_dense18/MatMul?
 5_dense18/BiasAdd/ReadVariableOpReadVariableOp)dense18_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype02"
 5_dense18/BiasAdd/ReadVariableOp?
5_dense18/BiasAddBiasAdd5_dense18/MatMul:product:0(5_dense18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
5_dense18/BiasAddv
5_dense18/ReluRelu5_dense18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
5_dense18/Relu?
6_dense9/MatMul/ReadVariableOpReadVariableOp%dense9_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02 
6_dense9/MatMul/ReadVariableOp?
6_dense9/MatMulMatMul5_dense18/Relu:activations:0&6_dense9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
6_dense9/MatMul?
6_dense9/BiasAdd/ReadVariableOpReadVariableOp&dense9_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
6_dense9/BiasAdd/ReadVariableOp?
6_dense9/BiasAddBiasAdd6_dense9/MatMul:product:0'6_dense9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
6_dense9/BiasAdds
6_dense9/ReluRelu6_dense9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
6_dense9/Relu?
$output_mem_clf/MatMul/ReadVariableOpReadVariableOp-output_mem_clf_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02&
$output_mem_clf/MatMul/ReadVariableOp?
output_mem_clf/MatMulMatMul6_dense9/Relu:activations:0,output_mem_clf/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output_mem_clf/MatMul?
%output_mem_clf/BiasAdd/ReadVariableOpReadVariableOp.output_mem_clf_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%output_mem_clf/BiasAdd/ReadVariableOp?
output_mem_clf/BiasAddBiasAddoutput_mem_clf/MatMul:product:0-output_mem_clf/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output_mem_clf/BiasAdd?
output_mem_clf/SoftmaxSoftmaxoutput_mem_clf/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
output_mem_clf/Softmax?
IdentityIdentity output_mem_clf/Softmax:softmax:0!^1_dense18/BiasAdd/ReadVariableOp ^1_dense18/MatMul/ReadVariableOp!^2_dense32/BiasAdd/ReadVariableOp ^2_dense32/MatMul/ReadVariableOp!^3_dense64/BiasAdd/ReadVariableOp ^3_dense64/MatMul/ReadVariableOp!^4_dense32/BiasAdd/ReadVariableOp ^4_dense32/MatMul/ReadVariableOp!^5_dense18/BiasAdd/ReadVariableOp ^5_dense18/MatMul/ReadVariableOp ^6_dense9/BiasAdd/ReadVariableOp^6_dense9/MatMul/ReadVariableOp&^output_mem_clf/BiasAdd/ReadVariableOp%^output_mem_clf/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????	: : : : : : : : : : : : : : 2D
 1_dense18/BiasAdd/ReadVariableOp 1_dense18/BiasAdd/ReadVariableOp2B
1_dense18/MatMul/ReadVariableOp1_dense18/MatMul/ReadVariableOp2D
 2_dense32/BiasAdd/ReadVariableOp 2_dense32/BiasAdd/ReadVariableOp2B
2_dense32/MatMul/ReadVariableOp2_dense32/MatMul/ReadVariableOp2D
 3_dense64/BiasAdd/ReadVariableOp 3_dense64/BiasAdd/ReadVariableOp2B
3_dense64/MatMul/ReadVariableOp3_dense64/MatMul/ReadVariableOp2D
 4_dense32/BiasAdd/ReadVariableOp 4_dense32/BiasAdd/ReadVariableOp2B
4_dense32/MatMul/ReadVariableOp4_dense32/MatMul/ReadVariableOp2D
 5_dense18/BiasAdd/ReadVariableOp 5_dense18/BiasAdd/ReadVariableOp2B
5_dense18/MatMul/ReadVariableOp5_dense18/MatMul/ReadVariableOp2B
6_dense9/BiasAdd/ReadVariableOp6_dense9/BiasAdd/ReadVariableOp2@
6_dense9/MatMul/ReadVariableOp6_dense9/MatMul/ReadVariableOp2N
%output_mem_clf/BiasAdd/ReadVariableOp%output_mem_clf/BiasAdd/ReadVariableOp2L
$output_mem_clf/MatMul/ReadVariableOp$output_mem_clf/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?

?
H__inference_output_mem_clf_layer_call_and_return_conditional_losses_2332

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
(__inference_1_dense18_layer_call_fn_2221

inputs
unknown:	
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_1_dense18_layer_call_and_return_conditional_losses_15682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?&
?
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_1955
hst_jobs
dense18_1919:	
dense18_1921:
dense32_1924: 
dense32_1926: 
dense64_1929: @
dense64_1931:@
dense32_1934:@ 
dense32_1936: 
dense18_1939: 
dense18_1941:
dense9_1944:	
dense9_1946:	%
output_mem_clf_1949:	!
output_mem_clf_1951:
identity??!1_dense18/StatefulPartitionedCall?!2_dense32/StatefulPartitionedCall?!3_dense64/StatefulPartitionedCall?!4_dense32/StatefulPartitionedCall?!5_dense18/StatefulPartitionedCall? 6_dense9/StatefulPartitionedCall?&output_mem_clf/StatefulPartitionedCall?
!1_dense18/StatefulPartitionedCallStatefulPartitionedCallhst_jobsdense18_1919dense18_1921*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_1_dense18_layer_call_and_return_conditional_losses_15682#
!1_dense18/StatefulPartitionedCall?
!2_dense32/StatefulPartitionedCallStatefulPartitionedCall*1_dense18/StatefulPartitionedCall:output:0dense32_1924dense32_1926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_2_dense32_layer_call_and_return_conditional_losses_15852#
!2_dense32/StatefulPartitionedCall?
!3_dense64/StatefulPartitionedCallStatefulPartitionedCall*2_dense32/StatefulPartitionedCall:output:0dense64_1929dense64_1931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_3_dense64_layer_call_and_return_conditional_losses_16022#
!3_dense64/StatefulPartitionedCall?
!4_dense32/StatefulPartitionedCallStatefulPartitionedCall*3_dense64/StatefulPartitionedCall:output:0dense32_1934dense32_1936*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_4_dense32_layer_call_and_return_conditional_losses_16192#
!4_dense32/StatefulPartitionedCall?
!5_dense18/StatefulPartitionedCallStatefulPartitionedCall*4_dense32/StatefulPartitionedCall:output:0dense18_1939dense18_1941*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_5_dense18_layer_call_and_return_conditional_losses_16362#
!5_dense18/StatefulPartitionedCall?
 6_dense9/StatefulPartitionedCallStatefulPartitionedCall*5_dense18/StatefulPartitionedCall:output:0dense9_1944dense9_1946*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_6_dense9_layer_call_and_return_conditional_losses_16532"
 6_dense9/StatefulPartitionedCall?
&output_mem_clf/StatefulPartitionedCallStatefulPartitionedCall)6_dense9/StatefulPartitionedCall:output:0output_mem_clf_1949output_mem_clf_1951*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_output_mem_clf_layer_call_and_return_conditional_losses_16702(
&output_mem_clf/StatefulPartitionedCall?
IdentityIdentity/output_mem_clf/StatefulPartitionedCall:output:0"^1_dense18/StatefulPartitionedCall"^2_dense32/StatefulPartitionedCall"^3_dense64/StatefulPartitionedCall"^4_dense32/StatefulPartitionedCall"^5_dense18/StatefulPartitionedCall!^6_dense9/StatefulPartitionedCall'^output_mem_clf/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????	: : : : : : : : : : : : : : 2F
!1_dense18/StatefulPartitionedCall!1_dense18/StatefulPartitionedCall2F
!2_dense32/StatefulPartitionedCall!2_dense32/StatefulPartitionedCall2F
!3_dense64/StatefulPartitionedCall!3_dense64/StatefulPartitionedCall2F
!4_dense32/StatefulPartitionedCall!4_dense32/StatefulPartitionedCall2F
!5_dense18/StatefulPartitionedCall!5_dense18/StatefulPartitionedCall2D
 6_dense9/StatefulPartitionedCall 6_dense9/StatefulPartitionedCall2P
&output_mem_clf/StatefulPartitionedCall&output_mem_clf/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
hst_jobs
?

?
C__inference_1_dense18_layer_call_and_return_conditional_losses_1568

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
-__inference_output_mem_clf_layer_call_fn_2341

inputs
unknown:	
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_output_mem_clf_layer_call_and_return_conditional_losses_16702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?

?
H__inference_output_mem_clf_layer_call_and_return_conditional_losses_1670

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?[
?
__inference__wrapped_model_1550
hst_jobsI
7sequential_mlp_1_dense18_matmul_readvariableop_resource:	F
8sequential_mlp_1_dense18_biasadd_readvariableop_resource:I
7sequential_mlp_2_dense32_matmul_readvariableop_resource: F
8sequential_mlp_2_dense32_biasadd_readvariableop_resource: I
7sequential_mlp_3_dense64_matmul_readvariableop_resource: @F
8sequential_mlp_3_dense64_biasadd_readvariableop_resource:@I
7sequential_mlp_4_dense32_matmul_readvariableop_resource:@ F
8sequential_mlp_4_dense32_biasadd_readvariableop_resource: I
7sequential_mlp_5_dense18_matmul_readvariableop_resource: F
8sequential_mlp_5_dense18_biasadd_readvariableop_resource:H
6sequential_mlp_6_dense9_matmul_readvariableop_resource:	E
7sequential_mlp_6_dense9_biasadd_readvariableop_resource:	N
<sequential_mlp_output_mem_clf_matmul_readvariableop_resource:	K
=sequential_mlp_output_mem_clf_biasadd_readvariableop_resource:
identity??/sequential_mlp/1_dense18/BiasAdd/ReadVariableOp?.sequential_mlp/1_dense18/MatMul/ReadVariableOp?/sequential_mlp/2_dense32/BiasAdd/ReadVariableOp?.sequential_mlp/2_dense32/MatMul/ReadVariableOp?/sequential_mlp/3_dense64/BiasAdd/ReadVariableOp?.sequential_mlp/3_dense64/MatMul/ReadVariableOp?/sequential_mlp/4_dense32/BiasAdd/ReadVariableOp?.sequential_mlp/4_dense32/MatMul/ReadVariableOp?/sequential_mlp/5_dense18/BiasAdd/ReadVariableOp?.sequential_mlp/5_dense18/MatMul/ReadVariableOp?.sequential_mlp/6_dense9/BiasAdd/ReadVariableOp?-sequential_mlp/6_dense9/MatMul/ReadVariableOp?4sequential_mlp/output_mem_clf/BiasAdd/ReadVariableOp?3sequential_mlp/output_mem_clf/MatMul/ReadVariableOp?
.sequential_mlp/1_dense18/MatMul/ReadVariableOpReadVariableOp7sequential_mlp_1_dense18_matmul_readvariableop_resource*
_output_shapes

:	*
dtype020
.sequential_mlp/1_dense18/MatMul/ReadVariableOp?
sequential_mlp/1_dense18/MatMulMatMulhst_jobs6sequential_mlp/1_dense18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_mlp/1_dense18/MatMul?
/sequential_mlp/1_dense18/BiasAdd/ReadVariableOpReadVariableOp8sequential_mlp_1_dense18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_mlp/1_dense18/BiasAdd/ReadVariableOp?
 sequential_mlp/1_dense18/BiasAddBiasAdd)sequential_mlp/1_dense18/MatMul:product:07sequential_mlp/1_dense18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_mlp/1_dense18/BiasAdd?
sequential_mlp/1_dense18/ReluRelu)sequential_mlp/1_dense18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_mlp/1_dense18/Relu?
.sequential_mlp/2_dense32/MatMul/ReadVariableOpReadVariableOp7sequential_mlp_2_dense32_matmul_readvariableop_resource*
_output_shapes

: *
dtype020
.sequential_mlp/2_dense32/MatMul/ReadVariableOp?
sequential_mlp/2_dense32/MatMulMatMul+sequential_mlp/1_dense18/Relu:activations:06sequential_mlp/2_dense32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
sequential_mlp/2_dense32/MatMul?
/sequential_mlp/2_dense32/BiasAdd/ReadVariableOpReadVariableOp8sequential_mlp_2_dense32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_mlp/2_dense32/BiasAdd/ReadVariableOp?
 sequential_mlp/2_dense32/BiasAddBiasAdd)sequential_mlp/2_dense32/MatMul:product:07sequential_mlp/2_dense32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_mlp/2_dense32/BiasAdd?
sequential_mlp/2_dense32/ReluRelu)sequential_mlp/2_dense32/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_mlp/2_dense32/Relu?
.sequential_mlp/3_dense64/MatMul/ReadVariableOpReadVariableOp7sequential_mlp_3_dense64_matmul_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_mlp/3_dense64/MatMul/ReadVariableOp?
sequential_mlp/3_dense64/MatMulMatMul+sequential_mlp/2_dense32/Relu:activations:06sequential_mlp/3_dense64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2!
sequential_mlp/3_dense64/MatMul?
/sequential_mlp/3_dense64/BiasAdd/ReadVariableOpReadVariableOp8sequential_mlp_3_dense64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential_mlp/3_dense64/BiasAdd/ReadVariableOp?
 sequential_mlp/3_dense64/BiasAddBiasAdd)sequential_mlp/3_dense64/MatMul:product:07sequential_mlp/3_dense64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 sequential_mlp/3_dense64/BiasAdd?
sequential_mlp/3_dense64/ReluRelu)sequential_mlp/3_dense64/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
sequential_mlp/3_dense64/Relu?
.sequential_mlp/4_dense32/MatMul/ReadVariableOpReadVariableOp7sequential_mlp_4_dense32_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_mlp/4_dense32/MatMul/ReadVariableOp?
sequential_mlp/4_dense32/MatMulMatMul+sequential_mlp/3_dense64/Relu:activations:06sequential_mlp/4_dense32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
sequential_mlp/4_dense32/MatMul?
/sequential_mlp/4_dense32/BiasAdd/ReadVariableOpReadVariableOp8sequential_mlp_4_dense32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_mlp/4_dense32/BiasAdd/ReadVariableOp?
 sequential_mlp/4_dense32/BiasAddBiasAdd)sequential_mlp/4_dense32/MatMul:product:07sequential_mlp/4_dense32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_mlp/4_dense32/BiasAdd?
sequential_mlp/4_dense32/ReluRelu)sequential_mlp/4_dense32/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_mlp/4_dense32/Relu?
.sequential_mlp/5_dense18/MatMul/ReadVariableOpReadVariableOp7sequential_mlp_5_dense18_matmul_readvariableop_resource*
_output_shapes

: *
dtype020
.sequential_mlp/5_dense18/MatMul/ReadVariableOp?
sequential_mlp/5_dense18/MatMulMatMul+sequential_mlp/4_dense32/Relu:activations:06sequential_mlp/5_dense18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_mlp/5_dense18/MatMul?
/sequential_mlp/5_dense18/BiasAdd/ReadVariableOpReadVariableOp8sequential_mlp_5_dense18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_mlp/5_dense18/BiasAdd/ReadVariableOp?
 sequential_mlp/5_dense18/BiasAddBiasAdd)sequential_mlp/5_dense18/MatMul:product:07sequential_mlp/5_dense18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_mlp/5_dense18/BiasAdd?
sequential_mlp/5_dense18/ReluRelu)sequential_mlp/5_dense18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_mlp/5_dense18/Relu?
-sequential_mlp/6_dense9/MatMul/ReadVariableOpReadVariableOp6sequential_mlp_6_dense9_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02/
-sequential_mlp/6_dense9/MatMul/ReadVariableOp?
sequential_mlp/6_dense9/MatMulMatMul+sequential_mlp/5_dense18/Relu:activations:05sequential_mlp/6_dense9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2 
sequential_mlp/6_dense9/MatMul?
.sequential_mlp/6_dense9/BiasAdd/ReadVariableOpReadVariableOp7sequential_mlp_6_dense9_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype020
.sequential_mlp/6_dense9/BiasAdd/ReadVariableOp?
sequential_mlp/6_dense9/BiasAddBiasAdd(sequential_mlp/6_dense9/MatMul:product:06sequential_mlp/6_dense9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2!
sequential_mlp/6_dense9/BiasAdd?
sequential_mlp/6_dense9/ReluRelu(sequential_mlp/6_dense9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
sequential_mlp/6_dense9/Relu?
3sequential_mlp/output_mem_clf/MatMul/ReadVariableOpReadVariableOp<sequential_mlp_output_mem_clf_matmul_readvariableop_resource*
_output_shapes

:	*
dtype025
3sequential_mlp/output_mem_clf/MatMul/ReadVariableOp?
$sequential_mlp/output_mem_clf/MatMulMatMul*sequential_mlp/6_dense9/Relu:activations:0;sequential_mlp/output_mem_clf/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2&
$sequential_mlp/output_mem_clf/MatMul?
4sequential_mlp/output_mem_clf/BiasAdd/ReadVariableOpReadVariableOp=sequential_mlp_output_mem_clf_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential_mlp/output_mem_clf/BiasAdd/ReadVariableOp?
%sequential_mlp/output_mem_clf/BiasAddBiasAdd.sequential_mlp/output_mem_clf/MatMul:product:0<sequential_mlp/output_mem_clf/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%sequential_mlp/output_mem_clf/BiasAdd?
%sequential_mlp/output_mem_clf/SoftmaxSoftmax.sequential_mlp/output_mem_clf/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2'
%sequential_mlp/output_mem_clf/Softmax?
IdentityIdentity/sequential_mlp/output_mem_clf/Softmax:softmax:00^sequential_mlp/1_dense18/BiasAdd/ReadVariableOp/^sequential_mlp/1_dense18/MatMul/ReadVariableOp0^sequential_mlp/2_dense32/BiasAdd/ReadVariableOp/^sequential_mlp/2_dense32/MatMul/ReadVariableOp0^sequential_mlp/3_dense64/BiasAdd/ReadVariableOp/^sequential_mlp/3_dense64/MatMul/ReadVariableOp0^sequential_mlp/4_dense32/BiasAdd/ReadVariableOp/^sequential_mlp/4_dense32/MatMul/ReadVariableOp0^sequential_mlp/5_dense18/BiasAdd/ReadVariableOp/^sequential_mlp/5_dense18/MatMul/ReadVariableOp/^sequential_mlp/6_dense9/BiasAdd/ReadVariableOp.^sequential_mlp/6_dense9/MatMul/ReadVariableOp5^sequential_mlp/output_mem_clf/BiasAdd/ReadVariableOp4^sequential_mlp/output_mem_clf/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????	: : : : : : : : : : : : : : 2b
/sequential_mlp/1_dense18/BiasAdd/ReadVariableOp/sequential_mlp/1_dense18/BiasAdd/ReadVariableOp2`
.sequential_mlp/1_dense18/MatMul/ReadVariableOp.sequential_mlp/1_dense18/MatMul/ReadVariableOp2b
/sequential_mlp/2_dense32/BiasAdd/ReadVariableOp/sequential_mlp/2_dense32/BiasAdd/ReadVariableOp2`
.sequential_mlp/2_dense32/MatMul/ReadVariableOp.sequential_mlp/2_dense32/MatMul/ReadVariableOp2b
/sequential_mlp/3_dense64/BiasAdd/ReadVariableOp/sequential_mlp/3_dense64/BiasAdd/ReadVariableOp2`
.sequential_mlp/3_dense64/MatMul/ReadVariableOp.sequential_mlp/3_dense64/MatMul/ReadVariableOp2b
/sequential_mlp/4_dense32/BiasAdd/ReadVariableOp/sequential_mlp/4_dense32/BiasAdd/ReadVariableOp2`
.sequential_mlp/4_dense32/MatMul/ReadVariableOp.sequential_mlp/4_dense32/MatMul/ReadVariableOp2b
/sequential_mlp/5_dense18/BiasAdd/ReadVariableOp/sequential_mlp/5_dense18/BiasAdd/ReadVariableOp2`
.sequential_mlp/5_dense18/MatMul/ReadVariableOp.sequential_mlp/5_dense18/MatMul/ReadVariableOp2`
.sequential_mlp/6_dense9/BiasAdd/ReadVariableOp.sequential_mlp/6_dense9/BiasAdd/ReadVariableOp2^
-sequential_mlp/6_dense9/MatMul/ReadVariableOp-sequential_mlp/6_dense9/MatMul/ReadVariableOp2l
4sequential_mlp/output_mem_clf/BiasAdd/ReadVariableOp4sequential_mlp/output_mem_clf/BiasAdd/ReadVariableOp2j
3sequential_mlp/output_mem_clf/MatMul/ReadVariableOp3sequential_mlp/output_mem_clf/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
hst_jobs
?

?
"__inference_signature_wrapper_2029
hst_jobs
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallhst_jobsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_15502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????	: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
hst_jobs
?
?
-__inference_sequential_mlp_layer_call_fn_1708
hst_jobs
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallhst_jobsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_16772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????	: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
hst_jobs
?

?
C__inference_3_dense64_layer_call_and_return_conditional_losses_2252

inputs0
matmul_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
(__inference_3_dense64_layer_call_fn_2261

inputs
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_3_dense64_layer_call_and_return_conditional_losses_16022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
(__inference_4_dense32_layer_call_fn_2281

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_4_dense32_layer_call_and_return_conditional_losses_16192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
C__inference_4_dense32_layer_call_and_return_conditional_losses_2272

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_5_dense18_layer_call_fn_2301

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_5_dense18_layer_call_and_return_conditional_losses_16362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
B__inference_6_dense9_layer_call_and_return_conditional_losses_1653

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_1_dense18_layer_call_and_return_conditional_losses_2212

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
-__inference_sequential_mlp_layer_call_fn_2168

inputs
unknown:	
	unknown_0:
	unknown_1: 
	unknown_2: 
	unknown_3: @
	unknown_4:@
	unknown_5:@ 
	unknown_6: 
	unknown_7: 
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_16772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????	: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
hst_jobs1
serving_default_hst_jobs:0?????????	B
output_mem_clf0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:Ї
?M
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
		optimizer


signatures
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
*u&call_and_return_all_conditional_losses
v__call__
w_default_save_signature"?I
_tf_keras_network?I{"name": "sequential_mlp", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "sequential_mlp", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hst_jobs"}, "name": "hst_jobs", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "1_dense18", "trainable": true, "dtype": "float32", "units": 18, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "1_dense18", "inbound_nodes": [[["hst_jobs", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "2_dense32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "2_dense32", "inbound_nodes": [[["1_dense18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "3_dense64", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "3_dense64", "inbound_nodes": [[["2_dense32", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "4_dense32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "4_dense32", "inbound_nodes": [[["3_dense64", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "5_dense18", "trainable": true, "dtype": "float32", "units": 18, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "5_dense18", "inbound_nodes": [[["4_dense32", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "6_dense9", "trainable": true, "dtype": "float32", "units": 9, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "6_dense9", "inbound_nodes": [[["5_dense18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_mem_clf", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_mem_clf", "inbound_nodes": [[["6_dense9", 0, 0, {}]]]}], "input_layers": [["hst_jobs", 0, 0]], "output_layers": [["output_mem_clf", 0, 0]]}, "shared_object_id": 22, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 9]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 9]}, "float32", "hst_jobs"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "sequential_mlp", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hst_jobs"}, "name": "hst_jobs", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "1_dense18", "trainable": true, "dtype": "float32", "units": 18, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "1_dense18", "inbound_nodes": [[["hst_jobs", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "2_dense32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "2_dense32", "inbound_nodes": [[["1_dense18", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "3_dense64", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "3_dense64", "inbound_nodes": [[["2_dense32", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "4_dense32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "4_dense32", "inbound_nodes": [[["3_dense64", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "5_dense18", "trainable": true, "dtype": "float32", "units": 18, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "5_dense18", "inbound_nodes": [[["4_dense32", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "6_dense9", "trainable": true, "dtype": "float32", "units": 9, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "6_dense9", "inbound_nodes": [[["5_dense18", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Dense", "config": {"name": "output_mem_clf", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_mem_clf", "inbound_nodes": [[["6_dense9", 0, 0, {}]]], "shared_object_id": 21}], "input_layers": [["hst_jobs", 0, 0]], "output_layers": [["output_mem_clf", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 24}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
#_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "hst_jobs", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 9]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hst_jobs"}}
?	

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
*x&call_and_return_all_conditional_losses
y__call__"?
_tf_keras_layer?{"name": "1_dense18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "1_dense18", "trainable": true, "dtype": "float32", "units": 18, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["hst_jobs", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}, "shared_object_id": 25}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}}
?	

kernel
bias
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
*z&call_and_return_all_conditional_losses
{__call__"?
_tf_keras_layer?{"name": "2_dense32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "2_dense32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["1_dense18", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 18}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18]}}
?	

kernel
 bias
#!_self_saveable_object_factories
"regularization_losses
#trainable_variables
$	variables
%	keras_api
*|&call_and_return_all_conditional_losses
}__call__"?
_tf_keras_layer?{"name": "3_dense64", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "3_dense64", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["2_dense32", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?	

&kernel
'bias
#(_self_saveable_object_factories
)regularization_losses
*trainable_variables
+	variables
,	keras_api
*~&call_and_return_all_conditional_losses
__call__"?
_tf_keras_layer?{"name": "4_dense32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "4_dense32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["3_dense64", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?	

-kernel
.bias
#/_self_saveable_object_factories
0regularization_losses
1trainable_variables
2	variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "5_dense18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "5_dense18", "trainable": true, "dtype": "float32", "units": 18, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["4_dense32", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?	

4kernel
5bias
#6_self_saveable_object_factories
7regularization_losses
8trainable_variables
9	variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "6_dense9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "6_dense9", "trainable": true, "dtype": "float32", "units": 9, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["5_dense18", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 18}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18]}}
?	

;kernel
<bias
#=_self_saveable_object_factories
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "output_mem_clf", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output_mem_clf", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["6_dense9", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}}
"
	optimizer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
 5
&6
'7
-8
.9
410
511
;12
<13"
trackable_list_wrapper
?
0
1
2
3
4
 5
&6
'7
-8
.9
410
511
;12
<13"
trackable_list_wrapper
?

Blayers
Cnon_trainable_variables
Dlayer_regularization_losses
regularization_losses
Elayer_metrics
Fmetrics
trainable_variables
	variables
v__call__
w_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
": 	21_dense18/kernel
:21_dense18/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Glayers
Hnon_trainable_variables
Ilayer_regularization_losses
regularization_losses
Jlayer_metrics
Kmetrics
trainable_variables
	variables
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
":  22_dense32/kernel
: 22_dense32/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Llayers
Mnon_trainable_variables
Nlayer_regularization_losses
regularization_losses
Olayer_metrics
Pmetrics
trainable_variables
	variables
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
":  @23_dense64/kernel
:@23_dense64/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?

Qlayers
Rnon_trainable_variables
Slayer_regularization_losses
"regularization_losses
Tlayer_metrics
Umetrics
#trainable_variables
$	variables
}__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
": @ 24_dense32/kernel
: 24_dense32/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?

Vlayers
Wnon_trainable_variables
Xlayer_regularization_losses
)regularization_losses
Ylayer_metrics
Zmetrics
*trainable_variables
+	variables
__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
":  25_dense18/kernel
:25_dense18/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?

[layers
\non_trainable_variables
]layer_regularization_losses
0regularization_losses
^layer_metrics
_metrics
1trainable_variables
2	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	26_dense9/kernel
:	26_dense9/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?

`layers
anon_trainable_variables
blayer_regularization_losses
7regularization_losses
clayer_metrics
dmetrics
8trainable_variables
9	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%	2output_mem_clf/kernel
!:2output_mem_clf/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?

elayers
fnon_trainable_variables
glayer_regularization_losses
>regularization_losses
hlayer_metrics
imetrics
?trainable_variables
@	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
	ltotal
	mcount
n	variables
o	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 32}
?
	ptotal
	qcount
r
_fn_kwargs
s	variables
t	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 24}
:  (2total
:  (2count
.
l0
m1"
trackable_list_wrapper
-
n	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
p0
q1"
trackable_list_wrapper
-
s	variables"
_generic_user_object
?2?
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_2082
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_2135
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_1955
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_1994?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_sequential_mlp_layer_call_fn_1708
-__inference_sequential_mlp_layer_call_fn_2168
-__inference_sequential_mlp_layer_call_fn_2201
-__inference_sequential_mlp_layer_call_fn_1916?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_1550?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
hst_jobs?????????	
?2?
C__inference_1_dense18_layer_call_and_return_conditional_losses_2212?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_1_dense18_layer_call_fn_2221?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_2_dense32_layer_call_and_return_conditional_losses_2232?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_2_dense32_layer_call_fn_2241?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_3_dense64_layer_call_and_return_conditional_losses_2252?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_3_dense64_layer_call_fn_2261?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_4_dense32_layer_call_and_return_conditional_losses_2272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_4_dense32_layer_call_fn_2281?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_5_dense18_layer_call_and_return_conditional_losses_2292?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_5_dense18_layer_call_fn_2301?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_6_dense9_layer_call_and_return_conditional_losses_2312?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_6_dense9_layer_call_fn_2321?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_output_mem_clf_layer_call_and_return_conditional_losses_2332?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_output_mem_clf_layer_call_fn_2341?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_2029hst_jobs"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
C__inference_1_dense18_layer_call_and_return_conditional_losses_2212\/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????
? {
(__inference_1_dense18_layer_call_fn_2221O/?,
%?"
 ?
inputs?????????	
? "???????????
C__inference_2_dense32_layer_call_and_return_conditional_losses_2232\/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? {
(__inference_2_dense32_layer_call_fn_2241O/?,
%?"
 ?
inputs?????????
? "?????????? ?
C__inference_3_dense64_layer_call_and_return_conditional_losses_2252\ /?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????@
? {
(__inference_3_dense64_layer_call_fn_2261O /?,
%?"
 ?
inputs????????? 
? "??????????@?
C__inference_4_dense32_layer_call_and_return_conditional_losses_2272\&'/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? {
(__inference_4_dense32_layer_call_fn_2281O&'/?,
%?"
 ?
inputs?????????@
? "?????????? ?
C__inference_5_dense18_layer_call_and_return_conditional_losses_2292\-./?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_5_dense18_layer_call_fn_2301O-./?,
%?"
 ?
inputs????????? 
? "???????????
B__inference_6_dense9_layer_call_and_return_conditional_losses_2312\45/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????	
? z
'__inference_6_dense9_layer_call_fn_2321O45/?,
%?"
 ?
inputs?????????
? "??????????	?
__inference__wrapped_model_1550? &'-.45;<1?.
'?$
"?
hst_jobs?????????	
? "??<
:
output_mem_clf(?%
output_mem_clf??????????
H__inference_output_mem_clf_layer_call_and_return_conditional_losses_2332\;</?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????
? ?
-__inference_output_mem_clf_layer_call_fn_2341O;</?,
%?"
 ?
inputs?????????	
? "???????????
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_1955r &'-.45;<9?6
/?,
"?
hst_jobs?????????	
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_1994r &'-.45;<9?6
/?,
"?
hst_jobs?????????	
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_2082p &'-.45;<7?4
-?*
 ?
inputs?????????	
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_mlp_layer_call_and_return_conditional_losses_2135p &'-.45;<7?4
-?*
 ?
inputs?????????	
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_mlp_layer_call_fn_1708e &'-.45;<9?6
/?,
"?
hst_jobs?????????	
p 

 
? "???????????
-__inference_sequential_mlp_layer_call_fn_1916e &'-.45;<9?6
/?,
"?
hst_jobs?????????	
p

 
? "???????????
-__inference_sequential_mlp_layer_call_fn_2168c &'-.45;<7?4
-?*
 ?
inputs?????????	
p 

 
? "???????????
-__inference_sequential_mlp_layer_call_fn_2201c &'-.45;<7?4
-?*
 ?
inputs?????????	
p

 
? "???????????
"__inference_signature_wrapper_2029? &'-.45;<=?:
? 
3?0
.
hst_jobs"?
hst_jobs?????????	"??<
:
output_mem_clf(?%
output_mem_clf?????????