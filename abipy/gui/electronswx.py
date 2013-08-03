from __future__ import print_function, division

import wx
import abipy.gui.awx as awx

try:
    from wxmplot import PlotFrame
except ImportError:
    import warnings
    warnings.warn("Error while import wxmplot. Some features won't be available")

from abipy import abiopen
from abipy.tools import AttrDict
from abipy.electrons import ElectronBands


def showElectronDosFrame(parent, filepath):
    bands = ElectronBands.from_file(filepath)
    ElectronDosFrame(parent, bands, title="File: %s" % filepath).Show()


def showElectronBandsPlot(parent, filepath):
    bands = ElectronBands.from_file(filepath)
    bands.plot(title="File: %s" % filepath)


def showElectronJdosFrame(parent, filepath):
    bands = ElectronBands.from_file(filepath)
    ElectronJdosFrame(parent, bands).Show()


class DosPanel(awx.Panel):
    """Base class defining a panel with controls for specifying the DOS parameters (step, width)."""
    DEFAULT_WIDTH = 0.2
    DEFAULT_STEP = 0.1

    def __init__(self, parent, **kwargs):
        super(DosPanel, self).__init__(parent, -1, **kwargs)
        self.BuildUi()

    def BuildUi(self):
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        label = wx.StaticText(self, -1, "Broadening [eV]:")
        label.Wrap(-1)
        self.width_ctrl = wx.SpinCtrlDouble(self, id=-1, value=str(self.DEFAULT_WIDTH), min=self.DEFAULT_WIDTH/1000, inc=0.1)

        main_sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP | wx.BOTTOM | wx.LEFT, 5)
        main_sizer.Add(self.width_ctrl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        label = wx.StaticText(self, -1, "Mesh step [eV]:")
        label.Wrap(-1)
        self.step_ctrl = wx.SpinCtrlDouble(self, id=-1, value=str(self.DEFAULT_STEP), min=self.DEFAULT_STEP/1000, inc=0.1)

        main_sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP | wx.BOTTOM | wx.LEFT, 5)
        main_sizer.Add(self.step_ctrl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        self.SetSizerAndFit(main_sizer)

    def GetParams(self):
        """Return the parameters for the computation of the DOS in a `AttrDict`."""
        return AttrDict(
            width=float(self.width_ctrl.GetValue()),
            step=float(self.step_ctrl.GetValue()),
        )


class PhononDosPanel(DosPanel):
    """Panel for the specification of the phonon DOS parameters."""
    DEFAULT_WIDTH = 0.002
    DEFAULT_STEP = 0.001


class ElectronDosPanel(DosPanel):
    """Panel for the specification of the electron DOS parameters."""
    DEFAULT_WIDTH = 0.2
    DEFAULT_STEP = 0.1


class ElectronDosDialog(wx.Dialog):
    """Dialog that asks the user to enter the parameters for the electron DOS."""

    def __init__(self, parent, **kwargs):
        super(ElectronDosDialog, self).__init__(parent, -1, **kwargs)

        self.SetSize((250, 200))
        self.SetTitle("Select DOS parameters")

        vbox = wx.BoxSizer(wx.VERTICAL)

        self.panel = ElectronDosPanel(self)

        hbox = wx.BoxSizer(wx.HORIZONTAL)

        ok_button = wx.Button(self, wx.ID_OK, label='Ok')
        close_button = wx.Button(self, wx.ID_CANCEL, label='Cancel')

        hbox.Add(ok_button)
        hbox.Add(close_button, flag=wx.LEFT, border=5)

        vbox.Add(self.panel, proportion=1, flag=wx.ALL | wx.EXPAND, border=5)
        vbox.Add(hbox, flag=wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, border=10)

        self.SetSizerAndFit(vbox)

    def GetParams(self):
        """Return a parameters in a `AttrDict`."""
        return self.panel.GetParams()


class ElectronDosFrame(awx.Frame):
    """This frames allows the user to control and compute the Electron DOS."""
    def __init__(self, parent, bands, **kwargs):
        """
            Args:
                bands:
                    `ElectronBands` instance.
                parent:
                    WX parent window.
        """
        if "title" not in kwargs:
            kwargs["title"] = "Electron DOS"

        super(ElectronDosFrame, self).__init__(parent, id=-1, **kwargs)
        self.bands = bands

        self.BuildUi()

        # Store PlotFrames in this list.
        self._pframes = []

    def BuildUi(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        panel = awx.Panel(self, -1)

        self.dos_panel = ElectronDosPanel(panel)
        main_sizer.Add(self.dos_panel, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        dos_button = wx.Button(panel, -1, "Compute DOS")
        self.Bind(wx.EVT_BUTTON, self.OnClick, dos_button)

        self.replot_checkbox = wx.CheckBox(panel, id=-1, label="Replot")
        self.replot_checkbox.SetValue(True)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)

        hsizer.Add(dos_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        hsizer.Add(self.replot_checkbox, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        main_sizer.Add(hsizer, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        panel.SetSizerAndFit(main_sizer)

    def OnClick(self, event):
        p = self.dos_panel.GetParams()
        if p.step == 1234:
            awx.tetris_game()

        try:
            edos = self.bands.get_dos(step=p.step, width=p.width)
        except:
            awx.showErrorMessage(self)
            return

        tot_dos, tot_idos = edos.dos_idos()
        label = "$\sigma = %s, step = %s$" % (p.width, p.step)

        plotframe = None
        if self.replot_checkbox.GetValue() and len(self._pframes):
            plotframe = self._pframes[-1]

        if plotframe is None:
            plotframe = PlotFrame(parent=self)
            plotframe.plot(tot_dos.mesh, tot_dos.values, label=label, draw_legend=True)
            plotframe.Show()
            self._pframes.append(plotframe)
        else:
            plotframe.oplot(tot_dos.mesh, tot_dos.values, label=label, draw_legend=True)


class ElectronJdosPanel(awx.Panel):
    """
    This panel allows the user to specify the parameters for the 
    calculation of the Electron JDOS.
    """
    DEFAULT_STEP = 0.1
    DEFAULT_WIDTH = 0.2

    def __init__(self, parent, nsppol, mband):
        """
            Args:
                parent:
                    WX parent window.
                nsppol:
                    Number of spins.
                mband:
                    Maximum number of bands

        """
        super(ElectronJdosPanel, self).__init__(parent, -1)

        self.nsppol = nsppol
        self.mband = mband

        self.BuildUi()

    def BuildUi(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        sz1 = wx.BoxSizer(wx.HORIZONTAL)

        spin_label = wx.StaticText(self, -1, "Spin:")
        spin_label.Wrap(-1)
        self.spin_cbox = wx.ComboBox(self, id=-1, choices=map(str, range(self.nsppol)), style=wx.CB_READONLY)

        sz1.Add(spin_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP | wx.BOTTOM | wx.LEFT, 5)
        sz1.Add(self.spin_cbox, 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP | wx.BOTTOM | wx.LEFT, 5)

        label = wx.StaticText(self, -1, "Broadening [eV]:")
        label.Wrap(-1)
        self.width_ctrl = wx.SpinCtrlDouble(self, id=-1, value=str(self.DEFAULT_WIDTH), min=self.DEFAULT_WIDTH/1000, inc=0.1)

        sz1.Add(label, 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP | wx.BOTTOM | wx.LEFT, 5)
        sz1.Add(self.width_ctrl, 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP | wx.BOTTOM | wx.LEFT, 5)

        label = wx.StaticText(self, -1, "Mesh step [eV]:")
        label.Wrap(-1)
        self.step_ctrl = wx.SpinCtrlDouble(self, id=-1, value=str(self.DEFAULT_STEP), min=self.DEFAULT_STEP/1000, inc=0.1)

        sz1.Add(label, 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP | wx.BOTTOM | wx.LEFT, 5)
        sz1.Add(self.step_ctrl, 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP | wx.BOTTOM | wx.LEFT, 5)

        main_sizer.Add(sz1)

        start_label = wx.StaticText(self, -1, "Start:")
        start_label.Wrap(-1)
        stop_label = wx.StaticText(self, -1, "Stop:")
        stop_label.Wrap(-1)

        self.vstart = wx.ComboBox(self, id=-1, choices=map(str, range(self.mband)))
        self.vstop = wx.ComboBox(self, id=-1, choices=map(str, range(self.mband)))

        val_sizer = wx.GridSizer(rows=2, cols=2, vgap=5, hgap=5)
        val_sizer.AddMany((start_label, self.vstart))
        val_sizer.AddMany((stop_label, self.vstop))

        val_sb = wx.StaticBox(self, label="Valence bands")
        val_boxsizer = wx.StaticBoxSizer(val_sb, wx.VERTICAL)
        val_boxsizer.Add(val_sizer)

        start_label = wx.StaticText(self, -1, "Start:")
        start_label.Wrap(-1)
        stop_label = wx.StaticText(self, -1, "Stop:")
        stop_label.Wrap(-1)

        self.cstart = wx.ComboBox(self, id=-1, choices=map(str, range(self.mband)))
        self.cstop = wx.ComboBox(self, id=-1, choices=map(str, range(self.mband)))

        cond_sizer = wx.GridSizer(rows=2, cols=2, vgap=5, hgap=5)
        cond_sizer.AddMany((start_label, self.cstart))
        cond_sizer.AddMany((stop_label, self.cstop))

        cond_sb = wx.StaticBox(self, label="Conduction bands")
        cond_boxsizer = wx.StaticBoxSizer(cond_sb, wx.VERTICAL)
        cond_boxsizer.Add(cond_sizer)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.AddMany((val_boxsizer, cond_boxsizer))

        main_sizer.Add(hsizer, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)
        self.SetSizerAndFit(main_sizer)

    def GetParams(self):
        """Returns the parameter for the JDOS calculation in a `AttrDict`"""
        vstart = int(self.vstart.GetValue())
        vstop = int(self.vstop.GetValue())
        vrange = range(vstart, vstop)
        if vstart == vstop: vrange = vstart

        cstart = int(self.cstart.GetValue())
        cstop = int(self.cstop.GetValue())
        crange = range(cstart, cstop)
        if cstart == cstop: crange = cstart

        return AttrDict(
            vrange=vrange,
            crange=crange,
            spin=int(self.spin_cbox.GetValue()),
            step=float(self.step_ctrl.GetValue()),
            width=float(self.width_ctrl.GetValue()),
            cumulative=True,
        )


class ElectronJdosDialog(wx.Dialog):
    """Dialog that asks the user to enter the parameters for the electron JDOS."""

    def __init__(self, parent, nsppol, mband, **kwargs):
        """
            Args:
                parent:
                    Parent window.
                nsppol:
                    Number of spins.
                mband
                    Maximum number of bands.
        """
        super(ElectronJdosDialog, self).__init__(parent, -1, **kwargs)

        self.SetSize((250, 200))
        self.SetTitle("Select JDOS parameters")

        vbox = wx.BoxSizer(wx.VERTICAL)

        self.panel = ElectronJDosPanel(self, nsppol, mband)

        hbox = wx.BoxSizer(wx.HORIZONTAL)

        ok_button = wx.Button(self, wx.ID_OK, label='Ok')
        close_button = wx.Button(self, wx.ID_CANCEL, label='Cancel')

        hbox.Add(ok_button)
        hbox.Add(close_button, flag=wx.LEFT, border=5)

        vbox.Add(self.panel, proportion=1, flag=wx.ALL | wx.EXPAND, border=5)
        vbox.Add(hbox, flag=wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, border=10)

        self.SetSizerAndFit(vbox)

    def GetParams(self):
        """Return a parameters in a `AttrDict`."""
        return self.panel.GetParams()


class ElectronJdosFrame(awx.Frame):
    """
    Frame for the computation of the JDOS.
    """
    def __init__(self, parent, bands, **kwargs):
        """
            Args:
                parent:
                    WX parent window.
                bands:
                    `ElectronBands` instance.
        """
        if "title" not in kwargs:
            kwargs["title"] = "Electron JDOS"

        super(ElectronJdosFrame, self).__init__(parent, id=-1, **kwargs)
        self.bands = bands

        self.BuildUi()

    def BuildUi(self):
        self.jdos_panel = ElectronJdosPanel(self, self.bands.nsppol, self.bands.mband)

        jdos_button = wx.Button(self, -1, "Compute JDOS", (20, 100))
        self.Bind(wx.EVT_BUTTON, self.OnClick, jdos_button)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.jdos_panel, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)
        sizer.Add(jdos_button, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.SetSizerAndFit(sizer)

    def OnClick(self, event):
        p = self.jdos_panel.GetParams()
        try:
            self.bands.plot_jdosvc(vrange=p.vrange, crange=p.crange, step=p.step, width=p.width,
                                   cumulative=p.cumulative)
        except:
            awx.showErrorMessage(self)

