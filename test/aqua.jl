module AquaTest

using Test
using ProximalOPT
using Aqua

@testset "Aqua" begin
  Aqua.test_all(ProximalOPT)
end

end
