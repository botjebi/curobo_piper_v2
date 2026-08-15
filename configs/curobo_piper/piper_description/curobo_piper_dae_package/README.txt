curobo_piper DAE conversion package
===================================

Purpose
-------
Use the color/material appearance from the supplied curobo_piper_v2 DAE files while keeping the original curobo_piper link/joint structure and STL collision meshes.

Important findings
------------------
1. base_link.STL and link1.STL ... link5.STL are byte-for-byte identical between the two supplied mesh ZIP files.
2. The supplied DAE files do NOT reference external PNG/JPG textures. Their appearance is stored as COLLADA materials/effects (diffuse colors); link2 also contains TEXCOORD data, which is preserved unchanged.
3. A few DAE files contained repeated isolated one-triangle artifacts far away from the link. These were removed in the converted files.

Applied local-frame transforms (v2 DAE -> original curobo_piper link frame)
-----------------------------------------------------------------------
base_link : Rz(0),                 t = (0, 0, 0)
link1     : Rz(-pi/2),             t = (0, 0, 0)
link2     : Rz(-0.10095 rad),      t = (0, 0, 0)
link3     : Rz(+1.759 rad),        t = (0, 0, 0)
link4     : Rz(+pi),               t = (0, 0, 0)
link5     : Rz(+pi),               t = (0, 0, -0.0014165 m)

Installation
------------
Copy the included dae/ directory to:
  <your_package>/piper_description/meshes/dae/

Then either use piper_description_dae.urdf as a reference, or change only the visual mesh paths for base_link through link5 to:
  package://piper_description/meshes/dae/<link_name>.dae

Keep collision mesh paths pointing to the original STL files.

Validation note
---------------
The converted DAE and STL outlines align closely, but the DAE geometry is richer/slightly different in some local details (especially link4/link5) rather than being an exact re-triangulation of the STL. This is expected from the supplied v2 visual meshes. Collision should therefore remain on the STL meshes.
