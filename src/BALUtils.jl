module BALUtils

using LightGraphs
using StaticArrays
using LinearAlgebra

export Camera, pose, axisangle, BA, readbal, visibility_graph, restrict, writebal, center, AngleAxis, num_cameras, num_points, num_observations

function rodrigues_rotate(angle_axis :: AbstractArray, point :: AbstractArray)
    theta = norm(angle_axis)
    axis = normalize(angle_axis)
    point * cos(theta) + cross(axis, point) * sin(theta) + axis * dot(axis, point) * (1 - cos(theta))
end

"""
Rotation represented as a 3 vector who's norm is the angle to rotate around the
normalized vector.
"""
struct AngleAxis{T <: Number}
    angleaxis :: SVector{3, T}
end

function AngleAxis(mat :: Array{<: Number, 2})
    AngleAxis(axisangle(mat))
end

function Base.:*(v :: AngleAxis, x :: AbstractArray)
    rodrigues_rotate(v.angleaxis, x)
end

# from https://math.stackexchange.com/questions/382760/composition-of-two-axis-angle-rotations
function Base.:*(x :: AngleAxis, y :: AngleAxis)
    alpha = norm(x.angleaxis)
    beta = norm(y.angleaxis)
    ax = normalize(x.angleaxis)
    ay = normalize(y.angleaxis)
    gamma = 2 * acos(cos(alpha/2) * cos(beta/2) - sin(alpha/2) * sin(beta/2) * dot(ax, ay))
    axis = (sin(alpha/2) * cos(beta/2) * ax + cos(alpha/2) * sin(beta/2) * ay - sin(alpha/2) * sin(beta/2)  * cross(ax, ay)) / sin(gamma/2)
    AngleAxis(axis * gamma)
end

Base.transpose(v :: AngleAxis) = AngleAxis(-v.angleaxis)
Base.adjoint(v :: AngleAxis) = transpose(v)

struct Camera{T}
    pose :: SVector{3,T}
    rotation :: AngleAxis{T}
    intrinsics :: SVector{3,T}
end

"""
Construct a camera from a rotation, translation, instrinsics vector.
"""
function Camera(x :: AbstractArray)
    Camera(SVector{3,Float64}(x[4:6]), AngleAxis(SVector{3,Float64}(x[1:3])), SVector{3,Float64}(x[7:9]))
end

pose(c :: Camera) = c.pose
center(c :: Camera) = -(c.rotation' * pose(c))
Base.vec(c :: Camera) = vcat(c.pose, c.rotation.angleaxis, c.intrinsics)

function axisangle(x :: Array{Float64, 2})
    u = [x[3,2]-x[2,3], x[3,1]-x[1,3], x[2,1] - x[1,2]]
    angle = acos((trace(x)-1)/2)
    SVector{3,Float64}((u / norm(u) * 2 * sin(angle))...)
end

struct BA
    cameras :: Vector{Camera}
    observations :: Vector{Vector{Tuple{Int64,Float64,Float64}}}
    points :: Vector{SVector{3,Float64}}
end

function restrict(ba :: BA, inds; ignore_points=false, vis_thresh=2)
    cams = ba.cameras[inds]
    obs = ba.observations[inds]
    obs, points = if !ignore_points
        # get all points still visible
        ps = vcat(map(x -> map(y -> y[1], x), obs)...)
        # remove points visible by only one camera (or threshold)
        ps = filter(x -> count(i -> i == x, ps) >= vis_thresh, unique(ps))
        # remap point ids
        d = Dict(zip(ps, 1:length(ps)))
        points = ba.points[ps]
        obs = map(obs) do ob
            vs = []
            for (i, x, y) in ob
                # drop points
                if i in keys(d)
                    push!(vs, (d[i], x, y))
                end
            end
            vs
        end
        obs, points
    else
        obs, ba.points
    end
    BA(cams, obs, points)
end

num_cameras(ba :: BA) = length(ba.cameras)
num_points(ba :: BA) = length(ba.points)
num_observations(ba :: BA) = sum(length.(ba.observations))

function Base.show(io :: IO, ba :: BA)
    print(io, "Bundle adjustment problem with $(length(ba.cameras)) cameras, $(length(ba.points)) points, $(sum(length.(ba.observations))) observations")
end

function readbal(filename)
    if extension(filename) == "bbal"
        open(filename, "b") do f
            num_cameras = read(f, UInt64) |> ntoh
            num_points = read(f, UInt64) |> ntoh
            num_observations = read(f, UInt64) |> ntoh

            obs = map(1:num_cameras) do i
                nobs = read(f, UInt64) |> ntoh
                obs = Vector{Tuple{Int64,Float64,Float64}}()
                sizehint!(obs, nobs)
                sizehint!(obs[i], nobs)
                for j in 1:nobs
                    p = read(f, UInt64) |> ntoh
                    u = read(f, Float64) |> ntoh
                    v = read(f, Float64) |> ntoh
                    push!(obs, (p+1, (u, v)))
                end
                obs
            end

            cameras = map(1:num_cameras) do i
                ary = Array{Float64}(undef, 9)
                read!(io, ary)
                Camera(ntoh.(ary))
            end

            points = map(1:num_points) do i
                ary = Array{Float64}(undef, 3)
                read!(io, ary)
                SVector{3,Float64}(ntoh.(ary))
            end

            BA(cameras, obs, points)
        end
    else
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
            Camera(t)
        end

        points = map(1:num_points) do i
            t = map(1:3) do j
                parse(Float64, readline(f))
            end
            SVector{3,Float64}(t...)
        end

        BA(cameras, obs, points)
    end
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

function writebal(filepath :: AbstractString, ba :: BA)
    open(filepath, "w") do f
        write(f, "$(num_cameras(ba)) $(num_points(ba)) $(num_observations(ba))\n")
        for (cam, obs) in enumerate(ba.observations)
            for (point, x, y) in obs
                # zero indexed
                write(f, "$(cam - 1) $(point - 1) $x $y\n")
            end
        end
        for cam in ba.cameras
            for v in vec(cam)
                write(f, "$v ")
            end
            write(f, "\n")
        end
        for point in ba.points
            write(f, "$(point[1]) $(point[2]) $(point[3])\n")
        end
    end
end

end # module
