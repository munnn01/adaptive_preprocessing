"""Standard bitstream, one independent decoder, separately measured inverse map."""
from pathlib import Path
import subprocess
import tempfile
import time
from ..preprocessing import validate_clip
from ..v32.codec import V32Codec,Packet
from ..v32.roi import inverse_retarget
from ..v32.transport import pack_packet


class MeasuredCodec(V32Codec):
    def roundtrip(self,rgb,*,source_shape,task,grids=(),encoded=None):
        validate_clip(rgb); t,h,w,_=rgb.shape; std=self.standard
        if h%2 or w%2: raise ValueError('YUV420 requires even dimensions')
        pack_packet(b'validate',task=task,codec=self.codec,source_shape=source_shape,coded_shape=rgb.shape,fps=self.fps,grids=grids)
        before=time.perf_counter(); encode_seconds=0.
        if encoded is None:
            encoder,fmt,ext=('libx264','h264','264') if self.codec=='h264' else ('libx265','hevc','265')
            with tempfile.TemporaryDirectory(prefix='v33-encode-') as directory:
                path=Path(directory)/('clip.'+ext)
                cmd=[std.ffmpeg,'-nostdin','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(std.fps),'-i','pipe:0','-an','-c:v',encoder,'-preset',std.preset,'-qp',str(std.qp),'-threads',str(std.threads),'-pix_fmt','yuv420p']
                params=f'pools={std.threads}:frame-threads=1:log-level=error' if self.codec=='h265' else ''
                g=std.gop_settings
                if g:
                    cmd+=['-g',str(g['gop']),'-keyint_min',str(g['gop'])]
                    options=f'keyint={g["gop"]}:min-keyint={g["gop"]}:scenecut={40 if g["scenecut"] else 0}:bframes={g["bframes"]}:open-gop={int(not g["closed_gop"])}'
                    params=f'{params}:{options}' if params else options
                if params: cmd+=['-x265-params' if self.codec=='h265' else '-x264-params',params]
                cmd+=['-f',fmt,str(path)]
                start=time.perf_counter(); subprocess.run(cmd,input=rgb.tobytes(),stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,timeout=std.timeout); stream=path.read_bytes(); encode_seconds=time.perf_counter()-start
                self.last_encode_command=cmd[:-1]+['<temporary-stream>']
        else: stream=encoded.data
        if not stream: raise RuntimeError('empty elementary stream')
        wire=pack_packet(stream,task=task,codec=self.codec,source_shape=source_shape,coded_shape=rgb.shape,fps=self.fps,grids=grids)
        # Decode coded geometry independently; inverse grids are from the verified wire recipe.
        from ..v32.transport import unpack_packet
        recipe=unpack_packet(wire)
        plain=pack_packet(stream,task=task,codec=self.codec,source_shape=rgb.shape,coded_shape=rgb.shape,fps=self.fps)
        start=time.perf_counter(); decoded=super().decode_packet(plain).decoded; decode_seconds=time.perf_counter()-start
        inverse_seconds=0.
        if recipe.grids:
            start=time.perf_counter(); decoded=inverse_retarget(decoded,recipe.grids,recipe.source_shape); inverse_seconds=time.perf_counter()-start
        decoded.setflags(write=False)
        self.last_times=dict(encode_seconds=encode_seconds,decode_seconds=decode_seconds,inverse_seconds=inverse_seconds)
        return Packet(wire,decoded,rgb.shape,len(stream),len(wire)-len(stream),time.perf_counter()-before)
