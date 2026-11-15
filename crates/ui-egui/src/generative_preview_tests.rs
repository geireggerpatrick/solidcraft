use serde_json::json;
use solidcraft_engine::doc::naming;

use super::*;
use crate::dialogs::GenerativeLoad;
use crate::scenario::Harness;

/// A wall and a shelf in front of it, and the Generative Design window with the wall's front
/// held and the shelf's underside loaded.
fn wall_and_shelf() -> (Harness, Dialog) {
    let mut h = Harness::new();
    for (name, corner, size) in [("Wall", [0, 0, 0], [5, 40, 50]), ("Shelf", [20, 5, 50], [40, 30, 4])] {
        let params = json!({"length": size[0], "width": size[1], "height": size[2], "corner": corner, "operation": "new", "body_name": name});
        h.app.run("solid.box", params).unwrap();
    }
    let mut d = Dialog::for_command(&h.app, "generative.design").unwrap();
    d.inputs[0].items = vec![face(&h, "Wall", Vec3::new(5.0, 20.0, 25.0))];
    d.inputs[1].items = vec![face(&h, "Shelf", Vec3::new(40.0, 20.0, 50.0))];
    (h, d)
}

fn face(h: &Harness, body: &str, point: Vec3) -> Sel {
    let state = crate::open_meshing::session_world(&h.app.session);
    let b = state.body(body).unwrap();
    let name = naming::face_name_at_body(b, point).unwrap();
    let (index, _) = naming::face_on_body(b, &name, point).unwrap();
    Sel::Face { body: body.into(), index, point }
}

fn near(a: Vec3, b: Vec3) -> bool {
    (a - b).len() < 1e-9
}

/// The window draws the force as an arrow on the loaded face and the design space as the box
/// around the picked faces. A force along the face's outward normal pulls at it, one against
/// it presses on it; the second load has its own arrows.
#[test]
fn the_force_and_the_design_space_are_shown() {
    let (h, mut d) = wall_and_shelf();
    let p = preview(&h.app, &d).unwrap();
    let (lo, hi) = p.space.unwrap();
    assert!(near(lo, Vec3::new(5.0, 0.0, 0.0)) && near(hi, Vec3::new(60.0, 40.0, 50.0)), "{lo:?} {hi:?}");
    assert_eq!(p.held.len(), 1);
    assert!(near(p.held[0].0, Vec3::new(5.0, 20.0, 25.0)));
    // 100 N downwards on the shelf's underside: away from the shelf.
    assert_eq!(p.forces.len(), 1);
    let f = &p.forces[0];
    assert!(near(f.at, Vec3::new(40.0, 20.0, 50.0)) && near(f.dir, Vec3::new(0.0, 0.0, -1.0)), "{f:?}");
    assert!(!f.onto && f.amount == 100.0 && f.load == 0 && f.kind == 0);

    // Upwards, 250 N, it presses on the face; three numbers give any direction.
    let Kind::Generative { loads, .. } = &mut d.kind else { panic!() };
    (loads[0].amount, loads[0].direction) = ("250".into(), 4);
    let p = preview(&h.app, &d).unwrap();
    assert!(p.forces[0].onto && p.forces[0].amount == 250.0 && near(p.forces[0].dir, Vec3::new(0.0, 0.0, 1.0)));
    let Kind::Generative { loads, .. } = &mut d.kind else { panic!() };
    loads[0].vector = "0; 3; -4".into();
    let p = preview(&h.app, &d).unwrap();
    assert!(near(p.forces[0].dir, Vec3::new(0.0, 0.6, -0.8)), "{:?}", p.forces[0].dir);

    // A second load on the shelf's side, a moment of 4 Nm about +Y, and a third on its top, a
    // pressure, which acts against the face's normal whatever direction is chosen.
    d.inputs[generative_load_input(1)].items = vec![face(&h, "Shelf", Vec3::new(40.0, 35.0, 52.0))];
    d.inputs[generative_load_input(2)].items = vec![face(&h, "Shelf", Vec3::new(40.0, 20.0, 54.0))];
    let Kind::Generative { loads, .. } = &mut d.kind else { panic!() };
    loads.push(GenerativeLoad { kind: 1, amount: "4".into(), direction: 2, vector: String::new(), case: 1 });
    loads.push(GenerativeLoad { kind: 2, amount: "0,5".into(), direction: 0, vector: String::new(), case: 0 });
    let p = preview(&h.app, &d).unwrap();
    assert_eq!(p.forces.len(), 3);
    let f = &p.forces[1];
    assert!(f.load == 1 && f.kind == 1 && f.amount == 4.0 && near(f.dir, Vec3::new(0.0, 1.0, 0.0)), "{f:?}");
    let f = &p.forces[2];
    assert!(f.load == 2 && f.kind == 2 && f.amount == 0.5 && f.onto && near(f.dir, Vec3::new(0.0, 0.0, -1.0)), "{f:?}");
    assert_eq!(p.loaded.iter().map(|(load, _)| *load).collect::<Vec<_>>(), [0, 1, 2]);

    // The command gets each load as what it is, the second in a load case of its own.
    let commands = crate::dialogs::dialog_commands(&h.app, &d).unwrap();
    let (command, params) = &commands[0];
    assert_eq!(command, "generative.design");
    assert_eq!(params["force"], json!([0.0, 150.0, -200.0]), "{params}");
    assert_eq!(params["loads"][0]["moment"], json!([0.0, 4.0, 0.0]), "{params}");
    assert_eq!(params["loads"][0]["case"], 2, "{params}");
    assert_eq!(params["loads"][1]["pressure"], 0.5, "{params}");
    assert_eq!(params["loads"][1]["case"], 1, "{params}");
}

/// Each setting with a place in space shows there: Room around widens the box, Mirror puts a
/// plane through its middle, Manufacturing shows the way up or the axis of the cut, Detail the
/// size of a cell and Thinnest member its width, Clearance the room around a body to avoid.
#[test]
fn the_settings_show_what_they_change() {
    let (mut h, mut d) = wall_and_shelf();
    let p = preview(&h.app, &d).unwrap();
    assert!(p.mirrors.is_empty() && p.making.is_none() && p.keep_out.is_empty() && p.guided.is_empty());
    // Medium detail: 480 000 cells in 55 x 40 x 50 mm; the thinnest member two and a half cells.
    let cell = (55.0f64 * 40.0 * 50.0 / 480_000.0).cbrt();
    assert!((p.cell.unwrap() - cell).abs() < 1e-9 && (p.member.unwrap() - 2.5 * cell).abs() < 1e-9);

    let Kind::Generative { margin, mirror, making, wall, detail, .. } = &mut d.kind else { panic!() };
    (*margin, *mirror, *making, *wall, *detail) = ("10".into(), 2, 1, "4".into(), 0);
    let p = preview(&h.app, &d).unwrap();
    let (lo, hi) = p.space.unwrap();
    assert!(near(lo, Vec3::new(-5.0, -10.0, -10.0)) && near(hi, Vec3::new(70.0, 50.0, 60.0)), "{lo:?} {hi:?}");
    // Mirrored in Y: the plane y = 20 across the box.
    assert_eq!(p.mirrors.len(), 1);
    assert!(p.mirrors[0].iter().all(|c| (c.y - 20.0).abs() < 1e-9));
    // Printed +Z up: from the middle of the underside, as high as the box.
    let Some(Making::Print { from, up, height, .. }) = p.making else { panic!("{:?}", p.making) };
    assert!(near(from, Vec3::new(32.5, 20.0, -10.0)) && near(up, Vec3::new(0.0, 0.0, 1.0)) && (height - 70.0).abs() < 1e-9);
    assert!(p.member == Some(4.0) && p.cell.unwrap() > cell, "coarser cells, the member as typed");

    // Cut along X: the box's section at both ends.
    let Kind::Generative { making, clearance, .. } = &mut d.kind else { panic!() };
    (*making, *clearance) = (7, "3".into());
    h.app
        .run("solid.box", json!({"length": 10, "width": 10, "height": 10, "corner": [30, 15, 20], "operation": "new", "body_name": "Pipe"}))
        .unwrap();
    d.inputs[2].items = vec![Sel::Body { name: "Pipe".into() }];
    let p = preview(&h.app, &d).unwrap();
    let Some(Making::Cut { ends }) = p.making else { panic!("{:?}", p.making) };
    assert!(ends[0].iter().all(|c| (c.x + 5.0).abs() < 1e-9) && ends[1].iter().all(|c| (c.x - 70.0).abs() < 1e-9));
    assert_eq!(p.keep_out.len(), 1);
    assert!(near(p.keep_out[0].0, Vec3::new(27.0, 12.0, 17.0)) && near(p.keep_out[0].1, Vec3::new(43.0, 28.0, 33.0)));

    // A face that slides and one on a pin are shown as what they are, and the command gets them.
    d.inputs[4].items = vec![face(&h, "Pipe", Vec3::new(35.0, 20.0, 30.0))];
    d.inputs[5].items = vec![face(&h, "Pipe", Vec3::new(35.0, 15.0, 25.0))];
    let p = preview(&h.app, &d).unwrap();
    assert_eq!(p.guided.iter().map(|(turns, ..)| *turns).collect::<Vec<_>>(), [false, true]);
    let params = &crate::dialogs::dialog_commands(&h.app, &d).unwrap()[0].1;
    assert!(params["slide"][0]["body"] == "Pipe" && params["pin"][0]["body"] == "Pipe", "{params}");

    // A body as the design space is the space; another window shows nothing.
    d.inputs[3].items = vec![Sel::Body { name: "Pipe".into() }];
    let (lo, hi) = preview(&h.app, &d).unwrap().space.unwrap();
    assert!(near(lo, Vec3::new(30.0, 15.0, 20.0)) && near(hi, Vec3::new(40.0, 25.0, 30.0)));
    assert!(preview(&h.app, &Dialog::for_command(&h.app, "solid.shell").unwrap()).is_none());
}
