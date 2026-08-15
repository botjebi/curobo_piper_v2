from pathlib import Path
from lxml import etree
import numpy as np
import shutil, zipfile, math, json

SRC_DAE = Path('/mnt/data/mesh_cmp/b/meshes_v2_gpt/dae')
OLD_URDF = Path('/mnt/data/piper_description(20260812-234635).urdf')
OUT = Path('/mnt/data/curobo_piper_dae_package')
DAE_OUT = OUT / 'dae'

# v2 DAE local frame -> original curobo_piper STL/link local frame.
# These are rigid transforms inferred from the two URDF frame conventions
# and verified against the byte-identical STL references supplied in both mesh sets.
TRANSFORMS = {
    'base_link': {'rz': 0.0,          't': [0.0, 0.0, 0.0]},
    'link1':     {'rz': -math.pi/2,   't': [0.0, 0.0, 0.0]},
    'link2':     {'rz': -0.10095,     't': [0.0, 0.0, 0.0]},
    'link3':     {'rz': 1.759,        't': [0.0, 0.0, 0.0]},
    'link4':     {'rz': math.pi,      't': [0.0, 0.0, 0.0]},
    'link5':     {'rz': math.pi,      't': [0.0, 0.0, -0.0014165]},
}


def rz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]], dtype=float)


def transform_source_array(source, R, t, is_position):
    # source -> float_array + accessor. Transform accessor records while preserving stride.
    fa = source.find('{*}float_array')
    acc = source.find('{*}technique_common/{*}accessor')
    if fa is None or acc is None or not fa.text:
        return
    vals = np.fromstring(fa.text, sep=' ', dtype=float)
    stride = int(acc.get('stride', '1'))
    count = int(acc.get('count', str(len(vals)//stride)))
    offset = int(acc.get('offset', '0'))
    if stride < 3 or count <= 0:
        return
    # Accessor starts at offset floats into the array.
    end = offset + count * stride
    if end > len(vals):
        return
    rec = vals[offset:end].reshape(count, stride)
    xyz = rec[:, :3]
    if is_position:
        xyz2 = xyz @ R.T + t
    else:
        xyz2 = xyz @ R.T
        n = np.linalg.norm(xyz2, axis=1)
        good = n > 1e-15
        xyz2[good] /= n[good, None]
    rec[:, :3] = xyz2
    vals[offset:end] = rec.reshape(-1)
    # 9 significant decimal places is ample relative to source precision while keeping files manageable.
    fa.text = ' '.join(f'{v:.9g}' for v in vals)


def clean_one_triangle_artifacts(tree, ns):
    # Several supplied DAE files contain repeated isolated one-triangle objects far from the robot link
    # (e.g. Mesh.728 / Mesh.024 / Mesh.397 / Mesh.219). Remove any geometry whose total triangle count <= 1.
    root = tree.getroot()
    removable = set()
    for geom in root.xpath('//c:library_geometries/c:geometry', namespaces=ns):
        tri_count = 0
        for tri in geom.xpath('./c:mesh/c:triangles', namespaces=ns):
            tri_count += int(tri.get('count', '0'))
        if tri_count <= 1:
            removable.add(geom.get('id'))
    if not removable:
        return 0
    removed_instances = 0
    for ig in list(root.xpath('//c:instance_geometry', namespaces=ns)):
        gid = (ig.get('url') or '').lstrip('#')
        if gid in removable:
            node = ig.getparent()
            node.remove(ig)
            removed_instances += 1
            # Remove empty node if it has no geometry/controllers/child nodes left.
            if len(node.xpath('./c:instance_geometry|./c:instance_controller|./c:node', namespaces=ns)) == 0:
                parent = node.getparent()
                if parent is not None:
                    parent.remove(node)
    for geom in list(root.xpath('//c:library_geometries/c:geometry', namespaces=ns)):
        if geom.get('id') in removable:
            geom.getparent().remove(geom)
    return removed_instances


def convert_one(name, src, dst, theta, trans):
    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    tree = etree.parse(str(src), parser)
    root = tree.getroot()
    uri = root.nsmap.get(None)
    ns = {'c': uri}
    R = rz(theta)
    t = np.asarray(trans, dtype=float)

    removed = clean_one_triangle_artifacts(tree, ns)

    # Find POSITION sources through <vertices>, and NORMAL sources through triangle inputs.
    pos_source_ids = set()
    for inp in root.xpath('//c:vertices/c:input[@semantic="POSITION"]', namespaces=ns):
        pos_source_ids.add((inp.get('source') or '').lstrip('#'))
    normal_source_ids = set()
    for inp in root.xpath('//c:triangles/c:input[@semantic="NORMAL"]', namespaces=ns):
        normal_source_ids.add((inp.get('source') or '').lstrip('#'))

    for source in root.xpath('//c:source', namespaces=ns):
        sid = source.get('id')
        if sid in pos_source_ids:
            transform_source_array(source, R, t, True)
        elif sid in normal_source_ids:
            transform_source_array(source, R, np.zeros(3), False)

    # Add a concise provenance comment under <asset> without affecting rendering.
    asset = root.find('{*}asset')
    if asset is not None:
        asset.append(etree.Comment(
            f' Converted for curobo_piper link frame: Rz={theta:.12g} rad, t={trans}; removed {removed} isolated one-triangle artifact instance(s). '
        ))

    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(dst), encoding='utf-8', xml_declaration=True, pretty_print=True)
    return removed


def build_modified_urdf(src_urdf, dst_urdf):
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(src_urdf), parser)
    root = tree.getroot()
    changed=[]
    for link in root.findall('link'):
        name=link.get('name')
        if name not in TRANSFORMS:
            continue
        for vis in link.findall('visual'):
            mesh=vis.find('geometry/mesh')
            if mesh is not None:
                mesh.set('filename', f'package://piper_description/meshes/dae/{name}.dae')
                changed.append(name)
    tree.write(str(dst_urdf), encoding='utf-8', xml_declaration=True, pretty_print=True)
    return changed


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    DAE_OUT.mkdir(parents=True)
    removed={}
    for name,cfg in TRANSFORMS.items():
        removed[name]=convert_one(name, SRC_DAE/f'{name}.dae', DAE_OUT/f'{name}.dae', cfg['rz'], cfg['t'])
    changed=build_modified_urdf(OLD_URDF, OUT/'piper_description_dae.urdf')

    notes = f'''curobo_piper DAE conversion package
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
'''
    (OUT/'README.txt').write_text(notes, encoding='utf-8')
    (OUT/'transforms.json').write_text(json.dumps(TRANSFORMS, indent=2), encoding='utf-8')
    # Include conversion script for reproducibility.
    shutil.copy2(__file__, OUT/'build_curobo_dae.py')

    zpath=Path('/mnt/data/curobo_piper_dae_package.zip')
    if zpath.exists(): zpath.unlink()
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in OUT.rglob('*'):
            if p.is_file():
                z.write(p, arcname=str(Path('curobo_piper_dae_package')/p.relative_to(OUT)))
    print('OUT',OUT)
    print('ZIP',zpath)
    print('removed',removed)
    print('changed',changed)

if __name__=='__main__':
    main()
