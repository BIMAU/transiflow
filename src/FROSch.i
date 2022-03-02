%module(directors = "1") FROSch
%{
// Includes that are used in this file
#include "FROSchPreconditioner.hpp"
#include "HYMLS_Solver.hpp"
#include "HYMLS_Exception.hpp"
%}

%feature("director:except")
{
    if ($error != NULL) {
        throw Swig::DirectorMethodException();
    }
}
%exception
{
    try
    {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    catch(HYMLS::Exception &e)
    {
        PyErr_SetString(PyExc_Exception, e.what());
        SWIG_fail;
    }
    catch (Swig::DirectorException &e)
    {
        SWIG_fail;
    }
}

// Includes that are exported
%include "FROSchPreconditioner.hpp"
%include "HYMLS_Solver.hpp"

 // We have to specify the methods below manually, because SWIG can't convert an RCP to const.
%extend FROSch::Preconditioner
{
    Preconditioner(Teuchos::RCP<Epetra_RowMatrix> m)
    {
        return new FROSch::Preconditioner(m);
    }

    int ApplyInverse(Teuchos::RCP<Epetra_MultiVector> x, Teuchos::RCP<Epetra_MultiVector> y)
    {
        return self->ApplyInverse(*x, *y);
    }
}

%extend HYMLS::Solver
{
    Solver(Teuchos::RCP<Epetra_RowMatrix> m, FROSch::Preconditioner &o, Teuchos::RCP<Teuchos::ParameterList> p)
    {
        return new HYMLS::Solver(m, Teuchos::rcp(&o, false), p);
    }

    int ApplyInverse(Teuchos::RCP<Epetra_MultiVector> x, Teuchos::RCP<Epetra_MultiVector> y)
    {
        return self->ApplyInverse(*x, *y);
    }
}

%exception;
