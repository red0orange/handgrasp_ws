import argparse
import asyncio
import os
from tqdm import tqdm

from isaacsim import SimulationApp


async def convert(in_file, out_file, load_materials=False):
    # This import causes conflicts when global
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # setup converter and flags
    converter_context.ignore_materials = not load_materials
    # converter_context.ignore_animation = False
    # converter_context.ignore_cameras = True
    # converter_context.single_mesh = True
    # converter_context.smooth_normals = True
    # converter_context.preview_surface = False
    # converter_context.support_point_instancer = False
    # converter_context.embed_mdl_in_usd = False
    # converter_context.use_meter_as_world_unit = True
    # converter_context.create_world_as_default_root_prim = False
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, progress_callback, converter_context)
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success


def asset_convert(input_folder, output_folder):
    supported_file_formats = ["stl", "obj", "fbx"]
    print(f"\nConverting folder {input_folder} to {output_folder}")

    (result, models) = omni.client.list(input_folder)
    for i, entry in tqdm(enumerate(models), total=len(models), desc="Converting"):
        model = str(entry.relative_path)
        model_name = os.path.splitext(model)[0]
        model_format = (os.path.splitext(model)[1])[1:]
        # Supported input file formats
        if model_format in supported_file_formats:
            input_model_path = input_folder + "/" + model
            converted_model_path = output_folder + "/" + model_name + ".usd"
            if not os.path.exists(converted_model_path):
                status = asyncio.get_event_loop().run_until_complete(
                    convert(input_model_path, converted_model_path, True)
                )
                if not status:
                    print(f"ERROR Status is {status}")
                # print(f"---Added {converted_model_path}")


if __name__ == "__main__":
    kit = SimulationApp()

    import omni
    from omni.isaac.core.utils.extensions import enable_extension

    enable_extension("omni.kit.asset_converter")

    input_folder = "/home/red0orange/Projects/handgrasp_ws/0_Data/IsaacSimGraspEvaCache/obj_to_convert"
    output_folder = "/home/red0orange/Projects/handgrasp_ws/0_Data/IsaacSimGraspEvaCache/converted_usd"
    asset_convert(input_folder=input_folder, output_folder=output_folder)

    # cleanup
    kit.close()