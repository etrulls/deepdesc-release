require 'nn'
require 'cunn'
require 'image'

-- load image
local im = image.rgb2y( image.lena() )

-- must be [0,255]
im = im:mul(255)

-- input: [batch size, 1, 64, 64]
local patches = torch.FloatTensor(2,1,64,64)
patches[1] = im[{ {},{1,64},{1,64} }]:clone()
patches[2] = im[{ {},{101,164},{101,164} }]:clone()

-- load model and mean
local data = torch.load( 'models/CNN3_p8_n8_split4_073000.t7' )
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

-- get descriptor
outp = desc:forward( patches ):float()
fn = 'desc.t7'
torch.save( fn, outp )
print( 'Saved to "' .. fn .. '"' )

