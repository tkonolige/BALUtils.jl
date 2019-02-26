module BALUtils

using LightGraphs

export Camera, pose, axisangle, BA, read_bal, visibility_graph

struct Camera
    pose :: NTuple{3, Float64} # in axis-angle format?
    rotation :: NTuple{3, Float64}
    intrinsics :: NTuple{3, Float64}
end

pose(c :: Camera) = c.pose

function axisangle(x :: Array{Float64, 2})
    u = [x[3,2]-x[2,3], x[3,1]-x[1,3], x[2,1] - x[1,2]]
    angle = acos((trace(x)-1)/2)
    tuple((u / norm(u) * 2 * sin(angle))...)
end

struct BA
    cameras :: Vector{Camera}
    observations :: Vector{Vector{Tuple{Int64,Float64,Float64}}}
    points :: Vector{NTuple{3, Float64}}
end

function Base.show(io :: IO, ba :: BA)
    print(io, "Bundle adjustment problem with $(length(ba.cameras)) cameras, $(length(ba.points)) points, $(sum(length.(ba.observations))) observations")
end

function read_bal(filename)
    f = open(filename)
    line = readline(f)
    (num_cameras, num_points, num_observations) = map(x->parse(Int,x), split(line))

    obs = [Vector{Tuple{Int64,Float64,Float64}}() for _ in  1:num_cameras]

    for i in 1:num_observations
        (c_ind_, p_ind_, x_, y_) = split(readline(f))
        c_ind = parse(Int, c_ind_)+1
        p_ind = parse(Int, p_ind_)+1
        x = parse(Float64, x_)
        y = parse(Float64, y_)
        push!(obs[c_ind], (p_ind, x, y))
    end

    cameras = map(1:num_cameras) do i
        t = map(1:9) do j
            parse(Float64, readline(f))
        end
        Camera(tuple(t[4:6]...), tuple(t[1:3]...), tuple(t[7:9]...))
    end

    points = map(1:num_points) do i
        t = map(1:3) do j
            parse(Float64, readline(f))
        end
        tuple(t...)
    end

    BA(cameras, obs, points)
end

function visibility_graph(ba :: BA)
    g = SimpleGraph(length(ba.points) + length(ba.cameras))
    for (i, obs) in enumerate(ba.observations)
        for (j, _, _) in obs
            add_edge!(g, i, length(ba.cameras) + j)
        end
    end
    m = adjacency_matrix(g)
    (m * m)[1:length(ba.cameras), 1:length(ba.cameras)]
end

end # module
