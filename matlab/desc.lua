require 'nn'
require 'cunn'
require 'mattorch'

local cmd = torch.CmdLine()
cmd:option( '--model', '../models/CNN3_p8_n8_split4_073000.t7', 'Network model to use' )
local params = cmd:parse(arg)

local input = mattorch.load( 'patches.mat' )
local patches = input.patches:float()

-- load model and mean
local data = torch.load( params.model )
local desc = data.desc
local mean = data.mean
local std  = data.std

-- normalize
for i=1,patches:size(1) do
   patches[i] = patches[i]:add( -mean ):cdiv( std )
end

-- convert to cuda for processing on the GPU
patches = patches:cuda()
desc:cuda()

-- save descriptor
local p = desc:forward( patches )
mattorch.save( 'desc.mat', {x = p:double() } )

