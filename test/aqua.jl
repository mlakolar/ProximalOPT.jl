module AquaTest

using Test
using ProximalSolvers
using Aqua

@testset "Aqua" begin
  Aqua.test_all(ProximalSolvers)
end

end
