mutable struct SGD{LR <: Real, GC <: Real}
    lr::LR
    gclip::GC
end

function update!(w, g, p::SGD)
end
