--[[
    Evaluate a network over a test set.
--]]

require "image"
require "math"
require "nn"
require "optim"
require "paths"
require "string"
require "torch"
require "xlua"

require "SpatialPyramidPooling"

cmd = torch.CmdLine()
cmd:text("Options")

cmd:option("-model-path",     "load-alexnet-l1-l2.t7", "Model path.")
cmd:option("-test-set-dir",   "",                      "Test dataset directory.")
cmd:option("-test-maps-path", "L2-L1.t7",              "Test domainness maps.")
cmd:option("-test-batch",     32,                      "Test batch size.")
cmd:option("-img-size",       227,                     "Image size.")
cmd:option("-use-gpu",        false,                   "Use CUDA or CPU.")
cmd:option("-gpuid",          1,                       "Which GPU to test the model.")

opt = cmd:parse(arg or {})

torch.setdefaulttensortype("torch.FloatTensor")

if opt["use-gpu"] then
    require "cunn"
    require "cudnn"
    require "cutorch"
    cutorch.setDevice(opt["gpuid"])
end


function load_batch(images_paths, start, offset)
    images = torch.Tensor(offset - start + 1, 3, 256, 256)
    labels = torch.Tensor(offset - start + 1)

    i = 1
    for p = start, offset do
        local sample = images_paths[p]:split("\t")
        local img = image.load(paths.concat(opt["test-set-dir"], "images", sample[1]))
        local label = tonumber(sample[2])

        images[i] = image.scale(img, 256, 256)
        labels[i] = label

        i = i + 1
    end

    return {
        images=images,
        labels=labels,
    }
end


function preprocess_batch(X, width, height)
    local R = 123.68
    local G = 116.779
    local B = 103.939

    local X_out = torch.Tensor(X:size(1), 3, height, width)

    for i = 1, X_out:size(1) do

        local original = X[i]:clone()

        -- Center crop --
        local infx = math.floor((original:size(2) - width) / 2)
        local supx = infx + width
        local processed = image.crop(original, infx, infx, supx, supx)

        -- Mean Pixel Subtraction --
        processed = processed:index(1, torch.LongTensor{3, 2, 1}):float():mul(255.0)
        local mean_pixel = torch.FloatTensor({B, G, R}):view(3, 1, 1):expandAs(processed)
        processed:csub(mean_pixel)

        X_out[i] = processed
    end

    return X_out
end


local test_set_paths = paths.concat(opt["test-set-dir"], "images.txt")
local test_set_size = 0
local images_paths = {}
for line in io.lines(test_set_paths) do
    images_paths[test_set_size] = line
    test_set_size = test_set_size + 1
end

local maps = torch.load(opt["test-maps-path"])
local model = torch.load(opt["model-path"])
model:evaluate()

if opt["use-gpu"] then
    model = model:cuda()
end

local accuracy = 0.0

for b = 1, test_set_size, opt["test-batch"] do
    local offset = math.min(b + opt["test-batch"] - 1, test_set_size - 1)

    local batch = load_batch(images_paths, b, offset)

    local X = preprocess_batch(batch.images, opt["img-size"], opt["img-size"])
    local M = maps[{ {b, offset} }]
    local Y = batch.labels

    if opt["use-gpu"] then
        X = X:cuda()
        M = M:cuda()
        Y = Y:cuda()
    end

    local max, ind = torch.max(model:forward({X, M}), 2)

    accuracy = accuracy + torch.sum(torch.eq(ind:long(), Y:long()))
    xlua.progress(b, test_set_size)
end

print("Test set accuracy: %.2f" % (accuracy / test_set_size * 100))
