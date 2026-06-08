/**
 * Created by amandaghassaei on 5/6/17.
 *
 * FOLD export. Rewritten for three r184 (no THREE.Geometry): vertex coordinates
 * come from the current node positions via model.getPositionsAsync() (which maps
 * the GPU buffer on the WebGPU path); edges/faces come from the FOLD data.
 */

export async function saveFOLD(){

    var positions = await globals.model.getPositionsAsync();

    if (!positions || positions.length === 0) {
        globals.warn("No geometry to save.");
        return;
    }

    var scale = globals.exportScale != 1 ? globals.exportScale : 1;

    var filename = $("#foldFilename").val();
    if (filename == "") filename = globals.filename;

    var json = {
        file_spec: 1.1,
        file_creator: "Origami Simulator: http://git.amandaghassaei.com/OrigamiSimulator/",
        file_author: $("#foldAuthor").val(),
        file_classes: ["singleModel"],
        frame_title: filename,
        frame_classes: ["foldedForm"],
        frame_attributes: ["3D"],
        frame_unit: globals.foldUnits,
        vertices_coords: [],
        edges_vertices: [],
        edges_assignment: [],
        faces_vertices: []
    };

    for (var i=0;i<positions.length;i+=3){
        json.vertices_coords.push([positions[i]*scale, positions[i+1]*scale, positions[i+2]*scale]);
    }

    var useTriangulated = globals.triangulateFOLDexport;
    if (!globals.includeCurves) {
        var fold = globals.pattern.getFoldData(!useTriangulated);
    } else {
        var fold = globals.curvedFolding.getFoldData(!useTriangulated);
    }
    json.edges_vertices = fold.edges_vertices;
    var assignment = [];
    for (var i=0;i<fold.edges_assignment.length;i++){
        if (fold.edges_assignment[i] == "C") assignment.push("B");
        else assignment.push(fold.edges_assignment[i]);
    }
    json.edges_assignment = assignment;
    json.faces_vertices = fold.faces_vertices;

    if (globals.exportFoldAngle){
        json.edges_foldAngle = fold.edges_foldAngle;
    }

    var blob = new Blob([JSON.stringify(json, null, 4)], {type: 'application/octet-binary'});
    saveAs(blob, filename + ".fold");
}
