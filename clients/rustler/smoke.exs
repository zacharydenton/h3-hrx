defmodule H3.Native do
  @on_load :load_nif
  def load_nif do
    path = System.fetch_env!("H3_NIF") |> String.to_charlist()
    :erlang.load_nif(path, 0)
  end
  def open(_audio_checkpoint), do: :erlang.nif_error(:not_loaded)
  def generate(_model, _request, _id), do: :erlang.nif_error(:not_loaded)
  def ping(_model, _id), do: :erlang.nif_error(:not_loaded)
  def encode_audio(_model, _samples, _frames, _id), do: :erlang.nif_error(:not_loaded)
end
model = H3.Native.open(nil)
:ok = H3.Native.ping(model, 42)
receive do
  {:h3_result, 42, {:ok, {[], 0}}} -> IO.puts("Rustler → H3 Rust API → HRX: ok")
after
  10_000 -> raise "model worker timed out"
end

:ok = H3.Native.generate(model, %{
  prompt: "fox", width: -32, height: 32, frames: 5, steps: 2, seed: 7,
  references: [], refmods: [], upscale_width: 64, upscale_height: 64,
  upscale_steps: 4, upscale_denoise: 0.4, upscale_seed: nil
}, 43)
receive do
  {:h3_result, 43, {:error, reason}} when is_binary(reason) ->
    IO.puts("Rustler generation request validation: ok")
after
  10_000 -> raise "generation validation timed out"
end
