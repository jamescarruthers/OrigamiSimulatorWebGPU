/**
 * Created by amandaghassaei on 5/2/17.
 *
 * STL/OBJ export. Rewritten for three r184: `THREE.Geometry`/`Face3` were
 * removed, so we build the export mesh directly from the current node positions
 * (via model.getPositionsAsync(), which maps the GPU buffer on the WebGPU path)
 * and the geometry's triangle index, computing per-face normals ourselves. The
 * `{ vertices, faces }` shape is what dependencies/binary_stl_writer.js consumes.
 */

// Build { vertices:[Vector3], faces:[{a,b,c,normal}] } from a flat positions
// array (length numVerts*3) and a triangle index array, applying `scale`.
function buildExportGeo(positions, indexArray, scale){
    var vertices = [];
    for (var i = 0; i < positions.length; i += 3){
        vertices.push(new THREE.Vector3(positions[i]*scale, positions[i+1]*scale, positions[i+2]*scale));
    }
    var faces = [];
    var ab = new THREE.Vector3(), ac = new THREE.Vector3();
    for (var i = 0; i < indexArray.length; i += 3){
        var a = indexArray[i], b = indexArray[i+1], c = indexArray[i+2];
        ab.subVectors(vertices[b], vertices[a]);
        ac.subVectors(vertices[c], vertices[a]);
        var normal = new THREE.Vector3().crossVectors(ab, ac).normalize();
        faces.push({ a: a, b: b, c: c, normal: normal });
    }
    return { vertices: vertices, faces: faces };
}

export async function makeSaveGEO(doublesided){
    var positions = await globals.model.getPositionsAsync();
    var geometry = globals.model.getGeometry();
    var index = geometry && geometry.index ? geometry.index.array : null;

    if (!positions || positions.length === 0 || !index || index.length === 0) {
        globals.warn("No geometry to save.");
        return null;
    }

    var geo = buildExportGeo(positions, index, globals.exportScale/globals.scale);

    if (doublesided){
        var numFaces = geo.faces.length;
        for (var i = 0; i < numFaces; i++){
            var face = geo.faces[i];
            // reversed winding -> opposite normal
            geo.faces.push({ a: face.a, b: face.c, c: face.b, normal: face.normal.clone().negate() });
        }
    }

    return geo;
}

export async function saveSTL(){
    var geo = await makeSaveGEO(globals.doublesidedSTL);
    if (!geo) return;
    var data = [{geo: geo, offset:new THREE.Vector3(0,0,0), orientation:new THREE.Quaternion(0,0,0,1)}];
    var stlBin = geometryToSTLBin(data);
    if (!stlBin) return;
    var blob = new Blob([stlBin], {type: 'application/octet-binary'});
    var filename = $("#stlFilename").val();
    if (filename == "") filename = globals.filename;
    saveAs(blob, filename + ".stl");
}

export async function saveOBJ(){
    //custom export to be compatible with freeform origami
    var positions = await globals.model.getPositionsAsync();
    if (!positions || positions.length === 0) {
        globals.warn("No geometry to save.");
        return;
    }
    var scale = globals.exportScale/globals.scale;
    var vertices = [];
    for (var i = 0; i < positions.length; i += 3){
        vertices.push(new THREE.Vector3(positions[i]*scale, positions[i+1]*scale, positions[i+2]*scale));
    }

    var flatGeo, fold;
    if (!globals.includeCurves) {
        flatGeo = globals.pattern.getFoldData(false);
        fold = globals.pattern.getFoldData(false);
    } else {
        flatGeo = globals.curvedFolding.getFoldData(false);
        fold = globals.curvedFolding.getFoldData(false);
    }

    var obj = "#output from https://origamisimulator.org/\n";
    obj += "# "+ vertices.length + "vertices\n";
    for (var i=0;i<vertices.length;i++){
        var vertex = vertices[i];
        obj += "v " + vertex.x + " " + vertex.y + " " + vertex.z + "\n"
    }
    obj += "# uv texture coords\n";
    // first get bounds for normalization
    var min = [Infinity, Infinity];
    var max = [-Infinity, -Infinity];
    for (var i=0;i<flatGeo.vertices_coords.length;i++){
        var vertex = flatGeo.vertices_coords[i];
        if (vertex[0] < min[0]) min[0] = vertex[0];
        if (vertex[2] < min[1]) min[1] = vertex[2];
        if (vertex[0] > max[0]) max[0] = vertex[0];
        if (vertex[2] > max[1]) max[1] = vertex[2];
    }
    var scaleUV = max[0] - min[0];
    if (max[1] - min[1] > scaleUV) scaleUV = max[1] - min[1];
    for (var i=0;i<flatGeo.vertices_coords.length;i++){
        var vertex = flatGeo.vertices_coords[i];
        obj += "vt " + (vertex[0] - min[0]) / scaleUV + " " + (vertex[2] - min[1]) / scaleUV + "\n"
    }
    obj += "# "+ fold.faces_vertices.length + " faces\n";
    for (var i=0;i<fold.faces_vertices.length;i++){
        var face = fold.faces_vertices[i];//triangular faces
        obj += "f " + (face[0]+1) + "/" + (face[0]+1) + " " + (face[1]+1) + "/" + (face[1]+1)+ " " +
         (face[2]+1) + "/" + (face[2]+1) + "\n"
    }

    obj += "# "+ fold.edges_vertices.length + " edges\n";
    for (var i=0;i<fold.edges_vertices.length;i++){
        var edge = fold.edges_vertices[i];//triangular faces
        obj += "#e " + (edge[0]+1) + " " + (edge[1]+1) + " ";
        if (fold.edges_assignment[i] == "F") obj += 1;
        else if (fold.edges_assignment[i] == "B") obj += 0;
        else if (fold.edges_assignment[i] == "M") obj += 3;
        else if (fold.edges_assignment[i] == "V") obj += 2;
        else {
            console.log("don't know how to convert type " + fold.edges_assignment[i]);
            obj += 0;
        }
        //todo fold angle
        obj += " 0\n";
    }

    var blob = new Blob([obj], {type: 'application/octet-binary'});
    var filename = $("#objFilename").val();
    if (filename == "") filename = globals.filename;
    saveAs(blob, filename + ".obj");
}
